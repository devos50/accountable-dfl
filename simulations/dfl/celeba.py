from asyncio import ensure_future

from simulations.args import get_args
from simulations.dfl.conflux_simulation import DFLSimulation

if __name__ == "__main__":
    args = get_args("celeba", default_lr=0.001)
    simulation = DFLSimulation(args)
    ensure_future(simulation.run())
    simulation.loop.run_forever()
