import numpy as np
import scipy

"""This class aims to have realistic variability models for AGN and Supernovae."""


class Variability(object):
    def __init__(self, variability_model, **kwargs_variability_model):
        """Initialize the variability class.

        :param variability_model: keyword for variability model to be used.
        :type variability_model: str
        :param kwargs_variability_model: Keyword arguments for variability class. For
            sinusoidal_variability kwargs are amplitude ('amp') and frequency ('freq').
        :type kwargs_variability_model: dict
        """
        self.variability_model = variability_model
        if self.variability_model not in ["sinusoidal"]:
            raise ValueError(
                "Given model is not supported. Currently supported model is sinusoidal."
            )
        if self.variability_model == "sinusoidal":
            self._model = sinusoidal_variability
        else:
            raise ValueError("Please provide a supported variability model.")

        self.kwargs_model = kwargs_variability_model

    def variability_at_time(self, observation_times):
        """Provides variability of a source at given time.

        :param observation_times: image observation time
        :return: variability at given time.
        """
        return self._model(observation_times, **self.kwargs_model)


def sinusoidal_variability(t, amp, freq):
    """Calculate the sinusoidal variability for a given observation time.

    :param t: observation time in [day].
    :param kwargs_model: dictionary of variability parameter associated with a source.
    :return: variability for the given time
    """
    return amp * np.sin(2 * np.pi * freq * t)


def interpolate_movie(Movie, Orig_timestamps, New_timestamps, verbose=False, plot=False):
    '''
    This function aims to take a series of snapshots and resample them at different time intervals.

    Movie is expected to be an array of (time, space_x, space_y),
    and Orig_timestamps is an array of timestamps representing the time values of each snapshot.
    New_timestamps is a list or array of new time stamps using the same units as Orig_timestamps.

    verbose allows some information to be printed to standard output.
    plot creates two plots, one with the original light curve plotted with the new interpolated light curve,
    and one where the second frame of each image is plotted side by side.

    returns an array representing the new resampled movie at timestamps New_timestamps, of shape
    (len(New_timestamps, np.size(Movie, 1), np.size(Movie, 2))
    '''

    initial_shape = np.shape(Movie)
    if verbose: print("The initial movie shape was",initial_shape)
    npix = initial_shape[1] * initial_shape[2]
    if verbose: print("This had",npix,"pixels")
    if verbose: print("Data will be interpolated to",len(New_timestamps), "frames")

    space_positions = np.linspace(1, npix, npix)                        # Define linear positions of pixels

    intermediate_movie = np.reshape(Movie, (initial_shape[0], npix))    # reshape image to a line

    interpolation = scipy.interpolate.RegularGridInterpolator((Orig_timestamps, space_positions),
                                            intermediate_movie, bounds_error=False, fill_value=None)

    new_timelength = np.size(New_timestamps)
    new_points_t, new_points_s = np.meshgrid(New_timestamps, space_positions, indexing='ij')
    movie_resampled = interpolation((new_points_t, new_points_s))
    if verbose: print("The new interpolated movie is now shape",np.shape(movie_resampled),"in time and spatial coordinates")

    # unpack to movie shape (time axis is and was first)
    resampled_movie = np.reshape(movie_resampled, (new_timelength, initial_shape[1], initial_shape[2])) 
    if verbose: print("The reconstructed movie is now shape",np.shape(resampled_movie))

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        ax.plot(np.asarray(Orig_timestamps), np.sum(Movie, axis=(1,2)), linewidth=2)
        ax.plot(np.asarray(New_timestamps), np.sum(resampled_movie, axis=(1,2)), '-o',alpha=0.5)

        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Flux [arb]")
        
        fig2, ax2 = plt.subplots(1,2)

        conts1 = ax2[0].contourf(Movie[1])
        conts2 = ax2[1].contourf(resampled_movie[1])

        ax2[0].set_xlabel("x [px]")
        ax2[1].set_xlabel("x [px]")
        ax2[0].set_ylabel("y [px]")
        ax2[1].set_ylabel("y [px]")
        ax2[0].set_title("Orig. movie, frame 2")
        ax2[1].set_title("Interp. movie, frame 2")

        plt.colorbar(conts1, ax=ax2[0], label="Flux [arb.]")
        plt.colorbar(conts2, ax=ax2[1], label="Flux [arb.]")


        for axis in ax2:
            axis.set_xlim(4*initial_shape[1]/9, 5*initial_shape[1]/9)
            axis.set_ylim(4*initial_shape[2]/9, 5*initial_shape[2]/9)
            axis.set_aspect(1)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
        
        plt.show()
    return resampled_movie   
