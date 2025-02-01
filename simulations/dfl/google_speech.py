from asyncio import ensure_future

from simulations.args import get_args
from simulations.dfl.conflux_simulation import ConfluxSimulation

if __name__ == "__main__":
    args = get_args("google_speech", default_lr=0.05)
    simulation = ConfluxSimulation(args)
    ensure_future(simulation.run())
    simulation.loop.run_forever()
