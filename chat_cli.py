#!/usr/bin/env python3
"""chat_cli.py - interactive client for the RAG chatbot.

Start the server first (in another terminal):

    make run

Then run this:

    uv run python chat_cli.py

Type messages and press Enter. 'exit' or Ctrl-C to quit.
"""

import itertools
import os

import requests

URL = os.getenv("CHAT_URL", "http://localhost:8000/chat")
session_id = next(itertools.count(1))


def main() -> None:
    print("RAG chat - type 'exit' or Ctrl-C to quit.")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg.lower() in {"exit", "quit"}:
            break
        try:
            r = requests.post(URL, json={"session_id": session_id, "message": msg})
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"error: {e}\n")
            continue
        print(f"bot> {r.json()['reply']}\n")


if __name__ == "__main__":
    main()
