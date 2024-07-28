import fire
import jax
from ring import utils
import tqdm
import tree_utils


def main(path_in, path_out, file_out_prefix: str = "seq"):

    data = utils.pickle_load(path_in)
    N = tree_utils.tree_shape(data)
    data = [jax.tree.map(lambda a: a[i], data) for i in range(N)]
    data = utils.replace_elements_w_nans(data, verbose=1)

    for i in tqdm.tqdm(range(N), total=N):
        file = utils.parse_path(
            path_out,
            file_out_prefix + str(i),
            extension="pickle",
            file_exists_ok=False,
        )
        utils.pickle_save(data[i], file)
        data[i] = None


if __name__ == "__main__":
    fire.Fire(main)
