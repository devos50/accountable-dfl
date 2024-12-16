import logging
from asyncio import ensure_future, get_event_loop

from simulations.args import get_args
from simulations.dfl.conflux_simulation import ConfluxSimulation

if __name__ == "__main__":
    args = get_args("cifar10", default_lr=0.002, default_momentum=0.9)
    simulation = ConfluxSimulation(args)

    async def main():
        try:
            await simulation.run()
        except Exception as e:
            logging.exception("An error occurred during simulation")
            get_event_loop().stop()

    task = ensure_future(main())
    simulation.loop.run_forever()
