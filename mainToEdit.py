"""
Ordinarily I think this code would be best organized into 1 class and several functions. As the code provided did not
split into classes I limited the code to putting every question in a function call. I thought this improved readability
over putting the answers in 3 .py files without substantially changing the structure as presented.
"""

"""
As the laptop I'll be using is weaker than my home computer, I've also screenshotted the outputs of the functional code
 in the event of a technical issue on the submission date
"""


# note Question 1

# note
def part1():
    """Answers to part 1
    Sine frequencies set to .2 and 20 for illustrative purposes"""

    # note The example says to put these in part 1. Ordinarily I would have them at the top of the code or class import
    # note as a result we end up importing these across the code
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    print("PyTorch Version:", torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]
    x = torch.Tensor(X)
    y = torch.Tensor(Y)

    x = x.to(device)
    y = y.to(device)

    gaussian = torch.exp(-(x ** 2 + y ** 2) / 2.0)

    sineafreq = .2
    sinebfreq = 20  # tried a few numbers here. Should mention this is a fixed number not a
    # variable which is generally undesirable

    # note 1a
    sine_2d = torch.sin(x * sineafreq) * torch.sin(y * sinebfreq)

    # note 1b
    gabor_filter = sine_2d * gaussian

    # plt.imshow(gaussian.cpu().numpy())

    plt.imshow(sine_2d.cpu().numpy())
    # plt.imshow(gabor_filter.cpu().numpy()) #not going to split 1) Into 2 sections
    plt.tight_layout()
    plt.show()


def part2_a():
    import torch
    import numpy as np

    # note This was not in the demo however I chose to make my code modular with functions
    # note so it is necessary
    # note alternately one could create a class and have a self.xxx reference
    # note as is I'm just re-inputting the device
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]

    # note Next line Commented out and overwritten - repalced with new values below
    # Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]

    # note I mislike the above - I would prefer to use fixed variables and input them so I do that below
    # mgrid - Start stop step
    center_x, center_y = -0.735, 0.2
    width, height = .5, .5
    spacing = 0.0001
    zoom = 30

    Y, X = np.mgrid[center_y - height / zoom:center_y + height / zoom:spacing,
           center_x - width / zoom:center_x + width / zoom:spacing]

    # load into PyTorch tensors
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    z = torch.complex(x, y)  # important!
    zs = z.clone()  # Updated!
    ns = torch.zeros_like(z)

    # transfer to the GPU device
    z = z.to(device)
    zs = zs.to(device)
    ns = ns.to(device)

    # Mandelbrot Set
    for i in range(200):
        # Compute the new values of z: z^2 + x
        zs_ = zs * zs + z
        # Have we diverged with this new value?
        not_diverged = torch.abs(zs_) < 4.0
        # Update variables to compute
        ns += not_diverged
        zs = zs_

    # plot
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 10))

    def processFractal(a):
        """Display an array of iteration counts as a
        colorful picture of a fractal."""

        a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
        img = np.concatenate([10 + 20 * np.cos(a_cyclic),
                              30 + 50 * np.sin(a_cyclic),
                              155 - 80 * np.cos(a_cyclic)], 2)
        img[a == a.max()] = 0
        a = img
        a = np.uint8(np.clip(a, 0, 255))
        return a

    plt.imshow(processFractal(ns.cpu().numpy()))
    plt.tight_layout(pad=0)
    plt.show()


def part2_b():
    import torch
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
    Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]

    # load into PyTorch tensors
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    z = torch.complex(x, y)  # initial values for Julia set
    ns = torch.zeros_like(z)

    # Define a fixed complex number for the Julia set
    c_real = torch.tensor(-0.7).to(device)
    c_imag = torch.tensor(0.27015).to(device)
    c_julia = torch.complex(c_real, c_imag)

    c_julia = c_julia.to(device)

    # transfer to the GPU device
    z = z.to(device)
    ns = ns.to(device)

    # Julia Set
    for i in range(200):
        # Compute the new values of z for Julia set: z^2 + c_julia
        z = z * z + c_julia
        # Check for divergence
        not_diverged = torch.abs(z) < 4.0
        # Update the iteration count for diverging points
        ns += not_diverged

    # plot
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 10))

    def processFractal(a):
        """Display an array of iteration counts as a
        colorful picture of a fractal."""
        a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
        img = np.concatenate([10 + 20 * np.cos(a_cyclic),
                              30 + 50 * np.sin(a_cyclic),
                              155 - 80 * np.cos(a_cyclic)], 2)
        img[a == a.max()] = 0
        a = img
        a = np.uint8(np.clip(a, 0, 255))
        return a

    plt.imshow(processFractal(ns.cpu().numpy()))
    plt.tight_layout(pad=0)
    plt.show()


def part3_a():
    """


    ## Lyapunov Fractal below #https://en.wikipedia.org/wiki/Lyapunov_fractal
    """

    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    # Configuration
    width, height = 800, 800
    r_min, r_max = 2.4, 4.0
    l_min, l_max = 2.4, 4.0
    iterations = 10000
    sequence = "RLRRRLRL"  # Example sequence, but you can choose any

    # Define the device: GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a meshgrid
    r = torch.linspace(r_min, r_max, width, device=device)
    l = torch.linspace(l_min, l_max, height, device=device)
    R, L = torch.meshgrid(r, l)

    # Initialize x and lambda (lyapunov exponent) to 0
    x = 0.5 * torch.ones(R.shape, device=device)
    lyapunov = torch.zeros(R.shape, device=device)

    # Iterate the function and calculate the Lyapunov exponent
    for i in range(iterations):
        if sequence[i % len(sequence)] == 'R':
            x = R * x * (1 - x)
            lyapunov += torch.log(
                torch.abs(R * (1 - 2 * x)))  # see the second part of the lambda expression in the wiki
        else:
            x = L * x * (1 - x)
            lyapunov += torch.log(torch.abs(L * (1 - 2 * x)))

    # Normalize the lyapunov values to be between -2 and 2 - better for plotting
    lyapunov = lyapunov / iterations
    lyapunov = torch.clip(lyapunov, -2, 2)

    # Move data back to CPU for visualization
    lyapunov = lyapunov.cpu()

    # # Display the fractal
    plt.imshow(lyapunov.numpy(), extent=(r_min, r_max, l_min, l_max), cmap='Accent')
    plt.title("Lyapunov Fractal")
    plt.show()

    pass


def part3_b():
    print("GITLINK HERE - GET LINKED")



part3_a()
