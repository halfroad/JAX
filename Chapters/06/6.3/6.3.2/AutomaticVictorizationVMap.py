import jax
import jax.numpy as jnp

def convolve(xs, ws):

    print("xs = {}\nws = {}".format(xs, ws))

    output = []

    for i in range(1, len(xs) - 1):

        dotted = jnp.dot(xs[i - 1: i + 2], ws)

        output.append(dotted)

    return jnp.array(output)

def start():

    x = jnp.arange(5)
    w = jnp.array([2., 3., 4.])

    xs = jnp.stack([x, x])
    ws = jnp.stack([w, w])

    array = convolve(xs, ws)

    # Empty
    print(array)

    auto_batch_convolve = jax.vmap(convolve)
    print(auto_batch_convolve(xs, ws))

    auto_batch_convolve_v2 = jax.vmap(convolve, in_axes = 1, out_axes = 1)

    xst = jnp.transpose(xs)
    wst = jnp.transpose(ws)

    print(auto_batch_convolve_v2(xst, wst))

    auto_batch_convolve_v3 = jax.vmap(convolve, in_axes = [0, None])
    print(auto_batch_convolve_v3(xs, w))

def main():

    start()

if __name__ == "__main__":

    main()
