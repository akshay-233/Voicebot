import asyncio
import json
import os
import threading
import queue
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, render_template
from flask_sock import Sock

from azure.core.credentials import AzureKeyCredential

from rtclient import RTClient, RealtimeException
from rtclient.models import NoTurnDetection  # keep it deterministic


load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

LIVE_SAMPLE_RATE = int(os.getenv("LIVE_SAMPLE_RATE", "24000"))

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_DEPLOYMENT in .env")
if not AZURE_OPENAI_API_KEY:
    raise RuntimeError("Missing AZURE_OPENAI_API_KEY in .env")


app = Flask(__name__)
sock = Sock(app)


@app.get("/")
def index():
    return render_template("index.html", sample_rate=LIVE_SAMPLE_RATE)


@dataclass
class OutMsg:
    kind: str  # "json" or "audio"
    data: bytes  # json bytes (utf-8) or raw pcm16 bytes


class RealtimeWorker:
    """
    Runs an asyncio loop in a background thread:
      - receives audio bytes via in_q (thread-safe queue)
      - forwards to RTClient.send_audio
      - on stop -> commit_audio -> generate_response -> stream out audio/text via out_q
    """

    def __init__(self, in_q: "queue.Queue[tuple[str, Optional[bytes]]]", out_q: "queue.Queue[OutMsg]"):
        self.in_q = in_q
        self.out_q = out_q
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        # unblock queue
        self.in_q.put(("__shutdown__", None))
        self._thread.join(timeout=2.0)

    def _thread_main(self):
        try:
            asyncio.run(self._async_main())
        except Exception as e:
            # send error to UI
            payload = json.dumps({"type": "error", "message": f"Worker crashed: {e}"}).encode("utf-8")
            self.out_q.put(OutMsg(kind="json", data=payload))

    async def _async_main(self):
        # Create RTClient session
        async with RTClient(
            url=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            key_credential=AzureKeyCredential(AZURE_OPENAI_API_KEY),
        ) as client:
            # Configure session: audio + text out, manual turn control
            await client.configure(
                modalities={"audio", "text"},
                turn_detection=NoTurnDetection(),
            )

            # Notify ready
            self.out_q.put(
                OutMsg(kind="json", data=json.dumps({"type": "status", "message": "Connected to Azure Realtime"}).encode("utf-8"))
            )

            # State
            collecting = False
            audio_sent = 0

            loop = asyncio.get_running_loop()

            async def q_get():
                # blockingly get from thread queue without blocking event loop
                return await loop.run_in_executor(None, self.in_q.get)

            while not self._stop_flag.is_set():
                msg_type, payload = await q_get()

                if msg_type == "__shutdown__":
                    break

                if msg_type == "start":
                    # Clear any server-side buffered audio if needed by committing empty? Best is clear_audio()
                    try:
                        await client.clear_audio()
                    except Exception:
                        pass
                    collecting = True
                    audio_sent = 0
                    self.out_q.put(
                        OutMsg(kind="json", data=json.dumps({"type": "status", "message": "Recording started"}).encode("utf-8"))
                    )

                elif msg_type == "audio" and collecting and payload:
                    try:
                        await client.send_audio(payload)
                        audio_sent += len(payload)
                    except Exception as e:
                        self.out_q.put(
                            OutMsg(kind="json", data=json.dumps({"type": "error", "message": f"send_audio failed: {e}"}).encode("utf-8"))
                        )

                elif msg_type == "stop":
                    collecting = False
                    self.out_q.put(
                        OutMsg(kind="json", data=json.dumps({"type": "status", "message": f"Recording stopped. Sent {audio_sent} bytes."}).encode("utf-8"))
                    )

                    # Commit audio (this creates an input_audio item server-side)
                    try:
                        input_item = await client.commit_audio()
                        await input_item  # wait for server processing of the audio item
                    except RealtimeException as e:
                        self.out_q.put(
                            OutMsg(kind="json", data=json.dumps({"type": "error", "message": f"commit_audio failed: {e.message}"}).encode("utf-8"))
                        )
                        continue
                    except Exception as e:
                        self.out_q.put(
                            OutMsg(kind="json", data=json.dumps({"type": "error", "message": f"commit_audio failed: {e}"}).encode("utf-8"))
                        )
                        continue

                    # Generate response (stream)
                    try:
                        response = await client.generate_response()
                    except Exception as e:
                        self.out_q.put(
                            OutMsg(kind="json", data=json.dumps({"type": "error", "message": f"generate_response failed: {e}"}).encode("utf-8"))
                        )
                        continue

                    # Stream items
                    try:
                        async for item in response:
                            # item.type can be "message" (text/audio) or others
                            if getattr(item, "type", None) == "message":
                                async for part in item:
                                    if part.type == "text":
                                        async for chunk in part.text_chunks():
                                            if chunk:
                                                self.out_q.put(
                                                    OutMsg(kind="json", data=json.dumps({"type": "text_delta", "delta": chunk}).encode("utf-8"))
                                                )
                                    elif part.type == "audio":
                                        async for chunk in part.audio_chunks():
                                            if chunk:
                                                # binary PCM16 bytes
                                                self.out_q.put(OutMsg(kind="audio", data=chunk))

                                        # transcript for the generated audio (optional)
                                        transcript_acc = ""
                                        async for tchunk in part.transcript_chunks():
                                            if tchunk:
                                                transcript_acc += tchunk
                                        if transcript_acc:
                                            self.out_q.put(
                                                OutMsg(kind="json", data=json.dumps({"type": "tts_transcript", "text": transcript_acc}).encode("utf-8"))
                                            )
                    except Exception as e:
                        self.out_q.put(
                            OutMsg(kind="json", data=json.dumps({"type": "error", "message": f"response stream failed: {e}"}).encode("utf-8"))
                        )

                    # mark done
                    self.out_q.put(
                        OutMsg(kind="json", data=json.dumps({"type": "done"}).encode("utf-8"))
                    )

                else:
                    # ignore unknown messages
                    pass


@sock.route("/ws")
def ws_handler(ws):
    """
    One websocket per browser tab.
    Receives:
      - JSON text frames: {"type":"start"} or {"type":"stop"}
      - Binary frames: PCM16LE @ 24000Hz audio chunks

    Sends:
      - JSON frames: status/errors/text deltas
      - Binary frames: PCM16LE audio chunks from model
    """
    in_q: "queue.Queue[tuple[str, Optional[bytes]]]" = queue.Queue()
    out_q: "queue.Queue[OutMsg]" = queue.Queue()

    worker = RealtimeWorker(in_q=in_q, out_q=out_q)
    worker.start()

    # Sender thread: pushes out_q → ws.send
    sender_stop = threading.Event()

    def sender():
        while not sender_stop.is_set():
            try:
                msg = out_q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if msg.kind == "json":
                    ws.send(msg.data.decode("utf-8"))
                elif msg.kind == "audio":
                    ws.send(msg.data)  # binary
            except Exception:
                break

    sender_thread = threading.Thread(target=sender, daemon=True)
    sender_thread.start()

    try:
        while True:
            data = ws.receive()
            if data is None:
                break

            # flask-sock returns `str` for text, `bytes` for binary
            if isinstance(data, str):
                try:
                    evt = json.loads(data)
                except Exception:
                    continue
                etype = evt.get("type")
                if etype == "start":
                    in_q.put(("start", None))
                elif etype == "stop":
                    in_q.put(("stop", None))
                else:
                    # unknown control message
                    pass

            elif isinstance(data, (bytes, bytearray)):
                # raw pcm16 bytes
                in_q.put(("audio", bytes(data)))

    finally:
        sender_stop.set()
        worker.stop()
        try:
            ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Flask dev server is OK for local testing.
    # For production, run behind a proper WSGI/ASGI server.
    app.run(host="0.0.0.0", port=5000, debug=True)
