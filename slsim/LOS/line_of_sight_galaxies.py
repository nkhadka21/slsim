import numpy as np
from slsim.lens import Lens
from slsim.lens_pop import draw_test_area

class LineofSightGalaxies(Lens):
    """Class to manage line of sight galxies for a lens.
    
    In this class, one should pass Galaxies class for the source_population and a 
    Deflector class for the deflector_class.
    """

    def __init__(
        self,
        source_population,
        deflector_class,
        cosmo,
        line_of_sight_area_factor=3,
        los_class=None,
        reference_area_range=None
    ):
        """
        :param source_population: A source population with in a certain sky area. 
         This must be a Galaxies class.
        :type source_class: Source class instance from slsim.Sources.source
        :param deflector_class: deflector instance
        :type deflector_class: Deflector class instance from slsim.Deflectors.deflector
        :param cosmo: astropy.cosmology instance
        :param test_area: area of disk around one lensing galaxies to be investigated
            on (in arc-seconds^2).
        :param los_class: line of sight dictionary (optional, takes these values instead of drawing from distribution)
        :type los_class: ~LOSIndividual() class object
        :param reference_area_range: The area in the lens image where random galaxies 
         need to be rendered. This is list or array of minimum and maximum area. 
         If provided a area is drawn uniformly from this range. Default is None.
        """
        vd = deflector_class.velocity_dispersion(cosmo=cosmo)
        if reference_area_range is None:
            self._test_area = line_of_sight_area_factor * draw_test_area(v_sigma=vd)
        else:
            self._test_area = np.random.uniform(reference_area_range[0],
                                                 reference_area_range[1])
        self._source_population = source_population
        self._z_max = deflector_class.redshift
        self.source_sample = self.draw_source_sample()
        if self.source_sample is not None:
            Lens.__init__(
            self,
            source_class=self.source_sample,
            deflector_class=deflector_class,
            cosmo=cosmo,
            test_area=self._test_area,
            los_class=los_class,
        )

    def get_num_line_of_sight_galaxies(self):
        """Computes the the number of galaxies in the line of sight for a given lens 
         test area.

        num_sources_tested_mean/ testarea = num_sources/ sky_area;
        testarea is in units of arcsec^2, f_sky is in units of deg^2. 1
        deg^2 = 12960000 arcsec^2
        """
        num_sources = self._source_population.source_number_selected
        num_galaxies_mean = (self._test_area * num_sources) / (
            12960000 * self._source_population.sky_area.to_value("deg2")
        )
        num_galaxies_range = np.random.poisson(lam=num_galaxies_mean)
        return num_galaxies_range
    
    def draw_source_sample(self):
        """This function draws sample of line of sight galaxies for a given lens"""
        num_galaxies = self.get_num_line_of_sight_galaxies()
        self._galaxies = []
        for _ in range(num_galaxies):
            galaxy=self._source_population.draw_source(z_max=self._z_max)
            #galaxy=self.draw_source()
            if galaxy is not None:
                self._galaxies.append(galaxy)
        if len(self._galaxies) == 1:
            galaxies = self._galaxies[0]
        elif len(self._galaxies) >= 2:
            galaxies=self._galaxies
        else:
            galaxies = None
        return galaxies



    def lenstronomy_kwargs(self, band=None):
        """Generates lenstronomy dictionary conventions for the class object.

        :param band: imaging band, if =None, will result in un-
            normalized amplitudes
        :type band: string or None
        :return: lenstronomy model and parameter conventions
        """
        # list of lens model. We provide empty list because we want to simulate unlensed
        #  galaxies within the certain aperture.
        kwargs_model = {
            "lens_light_model_list": [],
            "lens_model_list": [],
        }
        if self.source_sample is not None:
            sources, sources_kwargs = self.source_light_model_lenstronomy(band=band)
            # ensure that only the models that exist are getting added to kwargs_model
            for k in sources.keys():
                kwargs_model[k] = sources[k]

            kwargs_source = sources_kwargs["kwargs_source"]
            kwargs_ps = sources_kwargs["kwargs_ps"]
        #if there is no source at the line of sight, we provide empty list for everything.
        else:
            kwargs_source=[]
            kwargs_ps=[]
            kwargs_model["lens_light_model_list"] = []

        kwargs_params = {
            "kwargs_lens": [],
            "kwargs_source": kwargs_source,
            "kwargs_lens_light": [],
            "kwargs_ps": kwargs_ps,
        }

        return kwargs_model, kwargs_params
