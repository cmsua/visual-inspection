from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def finish_figure(
    figure: Figure,
    savefig: Optional[Union[str, Path]],
    show: bool,
    save_fig: Optional[Union[str, Path]] = None
) -> Optional[Figure]:
    destination = savefig if savefig is not None else save_fig
    if destination is not None:
        path = Path(destination).with_suffix('.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300)

    if show:
        plt.show()
        plt.close(figure)
        return None

    return figure
