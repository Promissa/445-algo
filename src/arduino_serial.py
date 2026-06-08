"""Serial client for the custom Arduino motion firmware.

Examples:
    python src/arduino_serial.py
    python src/arduino_serial.py ping
    python src/arduino_serial.py --port /dev/ttyACM0 init
    python src/arduino_serial.py command "MOVE X 1000 F 800 A 600"
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import serial
from serial.tools import list_ports

DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 2.0
INIT_DONE_TIMEOUT = 120.0
ARDUINO_PORT_KEYWORDS = (
    "arduino",
    "genuino",
    "usbmodem",
    "usbserial",
    "ttyacm",
    "ttyusb",
    "ttyama",      # Raspberry Pi GPIO UART (e.g. /dev/ttyAMA0)
    "ch340",
    "wch",
    "cp210",
    "silicon labs",
    "ftdi",
)


@dataclass
class ArduinoMotionClient:
    port: str
    baud: int = DEFAULT_BAUD
    timeout: float = DEFAULT_TIMEOUT
    reset_delay: float = 2.0

    def __post_init__(self) -> None:
        self.serial = serial.Serial(self.port, self.baud, timeout=self.timeout)
        if self.reset_delay > 0:
            time.sleep(self.reset_delay)
        self.drain()

    def close(self) -> None:
        self.serial.close()

    def drain(self) -> list[str]:
        lines: list[str] = []
        while self.serial.in_waiting:
            line = self._readline()
            if line:
                lines.append(line)
        return lines

    def send(self, command: str) -> None:
        self.serial.write((command.strip() + "\n").encode("ascii"))
        self.serial.flush()

    def command(self, command: str, timeout: float = DEFAULT_TIMEOUT) -> list[str]:
        self.send(command)
        return self.read_until_terminal(timeout=timeout)

    def read_until_terminal(self, timeout: float = DEFAULT_TIMEOUT) -> list[str]:
        lines: list[str] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._readline()
            if not line:
                continue
            lines.append(line)
            if is_terminal_response(line):
                break
        return lines

    def initialize(self, timeout: float = INIT_DONE_TIMEOUT) -> list[str]:
        self.send("INIT")
        lines: list[str] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._readline()
            if not line:
                continue
            lines.append(line)
            if line == "INIT DONE" or line.startswith("ERR "):
                break
        return lines

    def _readline(self) -> str:
        raw = self.serial.readline()
        return raw.decode("ascii", errors="replace").strip()


def is_terminal_response(line: str) -> bool:
    return (
        line in {"OK", "PONG", "INIT DONE"}
        or line.startswith("ERR ")
        or line.startswith("DONE ")
        or line.startswith("POS ")
        or line.startswith("LSX1=")
    )


def print_lines(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


def list_serial_ports() -> int:
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.", file=sys.stderr)
        return 1
    for port in ports:
        print(f"{port.device}\t{port.description}")
    return 0


def run_console(client: ArduinoMotionClient, timeout: float) -> None:
    print("Enter firmware commands. Type exit or quit to close.")
    while True:
        try:
            command = input("> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not command:
            continue
        if command.lower() in {"exit", "quit"}:
            break

        print_lines(client.command(command, timeout=timeout))


def find_arduino_port() -> str | None:
    ports = list(list_ports.comports())
    if not ports:
        return None

    scored_ports: list[tuple[int, str]] = []
    for port in ports:
        fields = (
            port.device,
            port.description or "",
            port.manufacturer or "",
            port.product or "",
            port.hwid or "",
        )
        searchable = " ".join(fields).lower()
        score = sum(1 for keyword in ARDUINO_PORT_KEYWORDS if keyword in searchable)
        if score:
            scored_ports.append((score, port.device))

    if scored_ports:
        scored_ports.sort(key=lambda item: (-item[0], item[1]))
        return scored_ports[0][1]

    if len(ports) == 1:
        return ports[0].device

    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control the Arduino motion firmware over serial."
    )
    parser.add_argument(
        "--port",
        help="Serial port, e.g. /dev/tty.usbmodemXXXX or COM3. Auto-detected when omitted.",
    )
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--no-reset-delay",
        action="store_true",
        help="Do not wait after opening serial.",
    )

    subparsers = parser.add_subparsers(dest="action")
    subparsers.add_parser("ports", help="List available serial ports.")
    subparsers.add_parser(
        "console", help="Continuously read commands from the console."
    )
    subparsers.add_parser("ping", help="Send PING.")
    subparsers.add_parser("status", help="Send STATUS?.")
    subparsers.add_parser("limits", help="Send LIMITS?.")
    subparsers.add_parser("init", help="Run INIT and wait for INIT DONE.")

    command_parser = subparsers.add_parser(
        "command", help="Send a raw firmware command."
    )
    command_parser.add_argument("command", help='Example: "MOVE X 1000 F 800 A 600"')

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    action = args.action or "console"

    if action == "ports":
        return list_serial_ports()

    port = args.port or find_arduino_port()
    if not port:
        parser.error(
            "could not auto-detect an Arduino serial port; pass --port or run the ports action"
        )

    client = ArduinoMotionClient(
        port=port,
        baud=args.baud,
        timeout=args.timeout,
        reset_delay=0.0 if args.no_reset_delay else 2.0,
    )
    try:
        if action != "init":
            print_lines(client.initialize())

        if action == "console":
            run_console(client, timeout=args.timeout)
        elif action == "ping":
            print_lines(client.command("PING", timeout=args.timeout))
        elif action == "status":
            print_lines(client.command("STATUS?", timeout=args.timeout))
        elif action == "limits":
            print_lines(client.command("LIMITS?", timeout=args.timeout))
        elif action == "init":
            print_lines(client.initialize())
        elif action == "command":
            print_lines(client.command(args.command, timeout=args.timeout))
        else:
            parser.error(f"unsupported action: {action}")
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
