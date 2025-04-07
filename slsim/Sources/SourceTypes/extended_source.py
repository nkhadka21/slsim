from slsim.Sources.SourceTypes.single_sersic import SingleSersic
from slsim.Sources.SourceTypes.double_sersic import DoubleSersic
from slsim.Sources.SourceTypes.interpolated_image import Interpolated

_SUPPORTED_EXTENDED_SOURCES = ["single_sersic", "double_sersic", "interpolated"]


class ExtendedSource(object):
    """Class to manage a single extended source."""

    def __init__(self, source_dict, cosmo=None, **kwargs):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         For a detailed description of this dictionary, please see the documentation for
         the SingleSersic, DoubleSersic, and Interpolated classes.
        :type source_dict: dict or astropy.table.Table
        :param extendedsource_type: keyword for specifying light profile model.
        :type extendedsource_type: str. supported types are "single_sersic",
         "double_sersic", "interpolated".
        :param cosmo: astropy.cosmology instance
        :param kwargs: dictionary of keyword arguments for a extended source.
         eg: kwargs = {"extendedsource_type": "single_sersic"}. Other supported
         types are "single_sersic", "double_sersic", "interpolated".
        """
        self.extendedsource_type = kwargs["extendedsource_type"]
        if self.extendedsource_type in ["single_sersic"]:
            self._source = SingleSersic(source_dict=source_dict)
        elif self.extendedsource_type in ["double_sersic"]:
            self._source = DoubleSersic(source_dict=source_dict)
        elif self.extendedsource_type in ["interpolated"]:
            self._source = Interpolated(source_dict=source_dict, cosmo=cosmo)
        else:
            raise ValueError(
                "Extended source type %s not supported. Chose among %s."
                % (self.extendedsource_type, _SUPPORTED_EXTENDED_SOURCES)
            )

    @property
    def redshift(self):
        """Returns source redshift."""

        return self._source.redshift

    def extended_source_position(self, reference_postion=None, draw_area=None):
        """Extended source position. If a center has already been provided (and
        stored in self._center_source during initialization of _source), then
        it is simply returned. Otherwise, a source position is drawn uniformly
        within the circle of the test area centered on the deflector position.
        see: _source.

        :param reference_position: reference position. the source postion will be
         defined relative to this position. The default choice is None. In this case 
         source_dict must contain source position.
         Eg: np.array([0, 0])
        :param draw_area: The area of the test region from which we randomly draw a source
         position. The default choice is None. In this case source_dict must contain 
         source position. Eg: 4*pi.
        :return: [x_pos, y_pos]
        """

        return self._source.extended_source_position(reference_postion, draw_area)

    def extended_source_magnitude(self, band):
        """Get the magnitude of the extended source in a specific band.

        :param band: Imaging band
        :type band: str
        :return: Magnitude of the extended source in the specified band
        :rtype: float
        """

        return self._source.extended_source_magnitude(band=band)

    def kwargs_extended_source_light(self, reference_position=None, draw_area=None, band=None):
        """Provides dictionary of keywords for the source light model(s).
        Kewords used are in lenstronomy conventions.

        :param reference_position: reference position. the source postion will be
         defined relative to this position. The default choice is None. In this case 
         source_dict must contain source position.
         Eg: np.array([0, 0])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. The default choice is None. In this case 
         source_dict must contain source position. Eg: 4*pi.
        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """

        return self._source.kwargs_extended_source_light(
            reference_position, draw_area, band
        )

    def extended_source_light_model(self):
        """Provides a list of source models.

        :return: list of extented source model.
        """
        return self._source.extended_source_light_model()

    def surface_brightness_reff(self, band=None):
        """Calculate average surface brightness within half light radius.

        :param band: Imageing band
        :return: average surface brightness within half light radius
            [mag/arcsec^2]
        """

        return self._source.surface_brightness_reff(band=band)
