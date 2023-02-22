from asyncio import ensure_future

from simulations.args import get_args
from simulations.dl.dl_simulation import DLSimulation

if __name__ == "__main__":
    args = get_args("celeba", default_lr=0.001)
    simulation = DLSimulation(args)
    ensure_future(simulation.run())
    simulation.loop.run_forever()
