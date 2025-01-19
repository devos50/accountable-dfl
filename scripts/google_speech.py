import asyncio
import logging

from scripts.run import run
from simulations.args import get_args

logging.basicConfig(level=logging.INFO)

async def main():
    args = get_args("google_speech", default_lr=0.05)
    await run(args)

if __name__ == "__main__":
    asyncio.run(main())
