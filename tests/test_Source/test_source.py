from slsim.Sources.source import Source, add_mean_mag_to_source_table
from slsim.Util.cosmo_util import z_scale_factor
import numpy as np
import pytest
from numpy import testing as npt
from astropy.table import Table
from astropy import cosmology


class TestSource:
    def setup_method(self):
        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict = Table(
            [
                [0.5],
                [17],
                [18],
                [16],
                [23],
                [24],
                [22],
                [0.5],
                [2],
                [4],
                [0.35],
                [0.8],
                [0.76],
                [20],
            ],
            names=(
                "z",
                "ps_mag_r",
                "ps_mag_g",
                "ps_mag_i",
                "mag_r",
                "mag_g",
                "mag_i",
                "amp",
                "freq",
                "n_sersic",
                "angular_size",
                "e1",
                "e2",
                "MJD",
            ),
        )
        source_dict2 = Table(
            [
                [0.5],
                [np.array([17, 18, 19, 20, 21])],
                [18],
                [16],
                [23],
                [24],
                [22],
                [0.5],
                [2],
                [4],
                [0.35],
                [0.8],
                [0.76],
                [0.001],
                [-0.001],
            ],
            names=(
                "z",
                "ps_mag_r",
                "ps_mag_g",
                "ps_mag_i",
                "mag_r",
                "mag_g",
                "mag_i",
                "amp",
                "freq",
                "n_sersic",
                "angular_size",
                "e1",
                "e2",
                "ra_off",
                "dec_off",
            ),
        )
        self.source_dict3 = Table(
            [
                [0.5],
                [1],
                [4],
                [0.0002],
                [0.00002],
                [0.01],
                [-0.02],
                [-0.002],
                [-0.023],
                [0.5],
                [0.5],
                [23],
                [0.001],
                [-0.001],
            ],
            names=(
                "z",
                "n_sersic_0",
                "n_sersic_1",
                "angular_size0",
                "angular_size1",
                "e0_1",
                "e0_2",
                "e1_1",
                "e1_2",
                "w0",
                "w1",
                "mag_i",
                "ra_off",
                "dec_off",
            ),
        )
        self.source_dict4 = Table(
            [
                [0.5],
                [1],
                [4],
                [0.0002],
                [0.00002],
                [0.01],
                [-0.02],
                [-0.002],
                [-0.023],
                [23],
                [0.001],
                [-0.001],
            ],
            names=(
                "z",
                "n_sersic_0",
                "n_sersic_1",
                "angular_size0",
                "angular_size1",
                "e0_1",
                "e0_2",
                "e1_1",
                "e1_2",
                "mag_i",
                "ra_off",
                "dec_off",
            ),
        )
        self.source_dict5 = {
            "angular_size": 0.1651633078964498,
            "center_x": 0.30298310338567075,
            "center_y": -0.3505004565139597,
            "e1": 0.06350855238708408,
            "e2": -0.08420760408362458,
            "mag_F106": 21.434711611915137,
            "mag_F129": 21.121205893763328,
            "mag_F184": 20.542431041034558,
            "n_sersic": 1.0,
            "z": 3.123,
        }

        self.source_dict5 = {
            "angular_size": 0.1651633078964498,
            "center_x": 0.30298310338567075,
            "center_y": -0.3505004565139597,
            "e1": 0.06350855238708408,
            "e2": -0.08420760408362458,
            "mag_F106": 21.434711611915137,
            "mag_F129": 21.121205893763328,
            "mag_F184": 20.542431041034558,
            "n_sersic": 1.0,
            "z": 3.123,
        }

        self.source = Source(
            self.source_dict,
            variability_model="sinusoidal",
            kwargs_variability={"amp", "freq"},
        )

        self.source2 = Source(
            source_dict2,
            variability_model="sinusoidal",
            kwargs_variability={"amp", "freq"},
        )
        self.source3 = Source(
            self.source_dict3,
            variability_model="light_curve",
            kwargs_variability={"supernovae_lightcurve", "i"},
            sn_absolute_mag_band="bessellb",
            sn_absolute_zpsys="ab",
            sn_type="Ia",
            lightcurve_time=np.linspace(-20, 50, 100),
            cosmo=cosmo,
            light_profile="double_sersic",
        )
        self.source4 = Source(
            self.source_dict3,
            variability_model="light_curve",
            kwargs_variability=None,
            sn_absolute_mag_band="bessellb",
            sn_absolute_zpsys="ab",
            sn_type="Ia",
            lightcurve_time=np.linspace(-20, 50, 100),
            cosmo=cosmo,
        )
        self.source5 = Source(
            self.source_dict4,
            variability_model="light_curve",
            kwargs_variability={"supernovae_lightcurve", "i"},
            sn_absolute_mag_band="bessellb",
            sn_absolute_zpsys="ab",
            sn_type="Ia",
            lightcurve_time=np.linspace(-20, 50, 100),
            cosmo=cosmo,
            light_profile="double_sersic",
        )

        self.source6 = Source(
            self.source_dict4,
            variability_model="light_curve",
            kwargs_variability={"supernovae_lightcurve", "F146", "z", "y"},
            sn_absolute_mag_band="bessellb",
            sn_absolute_zpsys="ab",
            sn_type="Ia",
            lightcurve_time=np.linspace(-20, 50, 100),
            cosmo=cosmo,
            light_profile="triplet",
        )

        self.source7 = Source(
            self.source_dict,
            variability_model="light_curve",
            kwargs_variability={"MJD", "ps_mag_r"},
        )
        self.source8 = Source(
            self.source_dict,
            variability_model="light_curve",
            kwargs_variability={"MJD", "ps_mag_z"},
        )
        self.source9 = Source(
            self.source_dict,
            variability_model="sinusoidal",
            kwargs_variability={"tmp", "fre"},
        )
        self.source10 = Source(
            self.source_dict3,
            variability_model="light_curve",
            kwargs_variability={"supernovae_lightcurve", "i"},
            sn_absolute_mag_band="bessellb",
            sn_absolute_zpsys="ab",
            sn_type="Ia",
            lightcurve_time=np.linspace(-20, 50, 100),
            cosmo=None,
        )
        self.source11 = Source(self.source_dict5, cosmo=cosmo)

        # Define AGN tests
        # try defining a specific light curve to use as the driving variability
        intrinsic_light_curve = {
            "MJD": np.linspace(1, 500, 500),
            "ps_mag_intrinsic": 10 + np.sin(np.linspace(1, 500, 500) * np.pi / 30),
        }

        # try to update the pool of masses which RandomAgn draws from
        agn_bounds_dict_update = {"black_hole_mass_exponent_bounds": (7.0, 8.0)}

        # define source dictionary for agn
        self.source_dict_agn_1 = Table(
            [
                [0.5],
                [4],
                [0.35],
                [0.8],
                [0.76],
                [[np.linspace(1, 500, 50)]],
                [20],
                [1000],
                [10],
                [500],
                [10],
                [9.5],
                [42],
            ],
            names=(
                "z",
                "n_sersic",
                "angular_size",
                "e1",
                "e2",
                "MJD",
                "ps_mag_i",
                "r_out",
                "corona_height",
                "r_resolution",
                "inclination_angle",
                "black_hole_mass_exponent",
                "random_seed",
            ),
        )

        self.source_dict_agn_2 = Table(
            [
                [0.5],
                [4],
                [0.35],
                [0.8],
                [0.76],
                [[np.linspace(1, 500, 50)]],
                [20],
                [1000],
                [10],
                [500],
                [10],
                [9.5],
                [agn_bounds_dict_update],
            ],
            names=(
                "z",
                "n_sersic",
                "angular_size",
                "e1",
                "e2",
                "MJD",
                "ps_mag_i",
                "r_out",
                "corona_height",
                "r_resolution",
                "inclination_angle",
                "black_hole_mass_exponent",
                "input_agn_bounds_dict",
            ),
        )

        self.source_dict_agn_no_mag = Table(
            [
                [0.5],
                [4],
                [0.35],
                [0.8],
                [0.76],
                [[np.linspace(1, 500, 50)]],
                [1000],
                [10],
                [500],
                [10],
                [9.5],
                [agn_bounds_dict_update],
            ],
            names=(
                "z",
                "n_sersic",
                "angular_size",
                "e1",
                "e2",
                "MJD",
                "r_out",
                "corona_height",
                "r_resolution",
                "inclination_angle",
                "black_hole_mass_exponent",
                "input_agn_bounds_dict",
            ),
        )

        # Create the agn source objects
        self.source_agn_1 = Source(
            self.source_dict_agn_1,
            variability_model="light_curve",
            kwargs_variability={
                "agn_lightcurve",
                "u",
                "g",
                "r",
                "i",
                "z",
                "y",
            },
            lightcurve_time=np.linspace(10, 500, 100),
            cosmo=cosmo,
        )

        self.source_agn_2 = Source(
            self.source_dict_agn_2,
            variability_model="light_curve",
            kwargs_variability={
                "agn_lightcurve",
                "u",
                "g",
                "r",
                "i",
                "z",
                "y",
            },
            lightcurve_time=np.linspace(-20, 500, 100),
            cosmo=cosmo,
            agn_driving_variability_model="light_curve",
            agn_driving_kwargs_variability=intrinsic_light_curve,
        )

        # test errors in source
        self.source_agn_error = Source(
            self.source_dict_agn_1,
            kwargs_variability={"agn_lightcurve"},
            lightcurve_time=np.linspace(-20, 50, 100),
            cosmo=cosmo,
        )

        self.source_agn_error_no_cosmo = Source(
            self.source_dict_agn_1,
            variability_model="light_curve",
            kwargs_variability={
                "agn_lightcurve",
                "u",
                "g",
                "r",
                "i",
                "z",
                "y",
            },
            lightcurve_time=np.linspace(10, 500, 100),
            cosmo=None,
        )

        self.source_agn_error_no_magnitude = Source(
            self.source_dict_agn_no_mag,
            variability_model="light_curve",
            kwargs_variability={
                "agn_lightcurve",
                "u",
                "g",
                "r",
                "i",
                "z",
                "y",
            },
            lightcurve_time=np.linspace(10, 500, 100),
            cosmo=cosmo,
        )

        self.source_agn_error_agn_not_in_kwargs = Source(
            self.source_dict3,
            variability_model="light_curve",
            kwargs_variability={"definitely_not_an_agn"},
            sn_absolute_mag_band="bessellb",
            sn_absolute_zpsys="ab",
            sn_type="Ia",
            lightcurve_time=np.linspace(-20, 50, 100),
            cosmo=cosmo,
        )

        # create an agn that has a broken power law driving signal
        self.source_dict_bpl_agn = Table(
            [
                [0.5],
                [4],
                [0.35],
                [0.8],
                [0.76],
                [[np.linspace(1, 500, 500)]],
                [20],
                [1000],
                [10],
                [500],
                [10],
                [9.5],
                [42],
            ],
            names=(
                "z",
                "n_sersic",
                "angular_size",
                "e1",
                "e2",
                "MJD",
                "ps_mag_i",
                "r_out",
                "corona_height",
                "r_resolution",
                "inclination_angle",
                "black_hole_mass_exponent",
                "random_seed",
            ),
        )
        # define bpl parameters
        variable_agn_kwarg_dict = {
            "length_of_light_curve": 250,
            "time_resolution": 1,
            "log_breakpoint_frequency": 1 / 20,
            "low_frequency_slope": 1,
            "high_frequency_slope": 3,
            "normal_magnitude_variance": 0.1,
        }
        # create agn source object with bpl driving variability
        self.source_bpl_agn = Source(
            self.source_dict_agn_1,
            variability_model="light_curve",
            kwargs_variability={
                "agn_lightcurve",
                "g",
                "i",
            },
            lightcurve_time=np.linspace(0, 500, 200),
            cosmo=cosmo,
            agn_driving_variability_model="bending_power_law",
            agn_driving_kwargs_variability=variable_agn_kwarg_dict,
        )

        # test errors when creating bpl (no input lightcurve_time)
        self.source_agn_bpl_error = Source(
            self.source_dict_agn_1,
            variability_model="light_curve",
            kwargs_variability={
                "agn_lightcurve",
                "g",
                "i",
            },
            cosmo=cosmo,
            agn_driving_variability_model="bending_power_law",
            agn_driving_kwargs_variability=variable_agn_kwarg_dict,
        )
        self.source_light_model_1 = Source(
            source_dict=self.source_dict,
            cosmo=cosmo,
            source_type="extended",
            light_profile="single_sersic",
        )
        self.source_light_model_2 = Source(
            source_dict=self.source_dict3,
            cosmo=cosmo,
            source_type="extended",
            light_profile="double_sersic",
        )
        self.source_light_model_3 = Source(
            source_dict=self.source_dict3,
            cosmo=cosmo,
            source_type="extended",
            light_profile="triple_sersic",
        )

        # Image Parameters
        size = 100
        center_brightness = 100
        noise_level = 10

        # Create a grid of coordinates
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        x, y = np.meshgrid(x, y)

        # Calculate the distance from the center
        r = np.sqrt(x**2 + y**2)

        # Create the galaxy image with light concentrated near the center
        image = center_brightness * np.exp(-(r**2) / 0.1)

        # Add noise to the image
        noise = noise_level * np.random.normal(size=(size, size))
        image += noise

        # Ensure no negative values
        image = np.clip(image, 0, None)
        test_image = image

        # Build a table for this "interp" source
        interp_source_dict1 = Table(
            names=(
                "z",
                "image",
                "center_x",
                "center_y",
                "z_data",
                "pixel_width_data",
                "phi_G",
                "mag_i",
                "mag_g",
                "mag_r",
            ),
            rows=[
                (
                    0.5,
                    test_image,
                    size // 2,
                    size // 2,
                    0.1,
                    0.05,
                    0.0,
                    20.0,
                    20.0,
                    20.0,
                )
            ],
        )

        interp_source_dict2 = Table(
            names=(
                "z",
                "image",
                "z_data",
                "pixel_width_data",
                "phi_G",
                "mag_i",
                "mag_g",
                "mag_r",
            ),
            rows=[
                (
                    0.5,
                    test_image,
                    0.1,
                    0.05,
                    0.0,
                    20.0,
                    20.0,
                    20.0,
                )
            ],
        )

        self.source_interp1 = Source(
            source_dict=interp_source_dict1,
            cosmo=cosmo,
            source_type="extended",
            light_profile="interpolated",
        )
        self.source_interp2 = Source(
            source_dict=interp_source_dict2,
            cosmo=cosmo,
            source_type="extended",
            light_profile="interpolated",
        )

    def test_kwargs_extended_source_light_interpolated(self):
        result = self.source_interp1.kwargs_extended_source_light(
            center_lens=np.array([0, 0]), draw_area=4 * np.pi, band="i"
        )
        source_array = self.source_interp1.source_dict["image"][0]
        size = source_array.shape[0]
        z = self.source_interp1.source_dict["z"][0]
        z_data = self.source_interp1.source_dict["z_data"][0]
        ratio = z_scale_factor(z_old=z_data, z_new=z, cosmo=self.source_interp1.cosmo)
        pixel_width_data = self.source_interp1.source_dict["pixel_width_data"][0]
        expected_scale = pixel_width_data * ratio

        assert result[0]["magnitude"] == 20.0
        npt.assert_allclose(
            result[0]["image"], source_array, rtol=1e-5, err_msg="Images differ!"
        )
        assert result[0]["center_x"] == size // 2
        assert result[0]["center_y"] == size // 2
        assert result[0]["phi_G"] == 0.0
        npt.assert_allclose(
            float(result[0]["scale"]),
            expected_scale,
            rtol=1e-4,
            err_msg="Pixel scale mismatch after z_scale_factor!",
        )

    def test_extended_source_light_model_interpolated(self):
        result = self.source_interp1.extended_source_light_model()
        assert result == ["INTERPOL"]

    def test_extended_source_position_interpolated(self):
        center_lens = np.array([0, 0])
        draw_area = 4 * np.pi

        # For source_interp1 (which has center_x and center_y provided),

        result_with_center = self.source_interp1.extended_source_position(
            center_lens, draw_area
        )
        expected_position_with_center = np.array(
            [
                self.source_interp1.source_dict["center_x"][0],
                self.source_interp1.source_dict["center_y"][0],
            ]
        )
        np.testing.assert_array_almost_equal(
            result_with_center.flatten(),
            expected_position_with_center,
            decimal=5,
            err_msg="Center position from source_dict does not match expected when center_x and center_y exist.",
        )

        # For source_interp2 (which does not have center_x/center_y),
        # force deletion of _center_source to trigger random position generation.
        if hasattr(self.source_interp2, "_center_source"):
            del self.source_interp2._center_source

        result_no_center = self.source_interp2.extended_source_position(
            center_lens, draw_area
        )
        test_area_radius = np.sqrt(draw_area / np.pi)
        r = np.linalg.norm(result_no_center - center_lens)
        assert (
            0 <= r <= test_area_radius
        ), f"Generated position {result_no_center} is out of expected test area range."

    def test_redshift(self):
        assert self.source.redshift == 0.5
        assert self.source11.redshift == 3.123

    def test_n_sersic(self):
        assert self.source.n_sersic == 4

    def test_angular_size(self):
        assert self.source.angular_size == 0.35

    def test_ellipticity(self):
        assert self.source.ellipticity[0] == 0.8
        assert self.source.ellipticity[1] == 0.76

    def test_ps_magnitude_no_variability(self):
        result = self.source.point_source_magnitude("r")
        assert result == [17]
        with pytest.raises(ValueError):
            self.source.point_source_magnitude("j")
        with pytest.raises(ValueError):
            self.source4.point_source_magnitude("r")
        result2 = self.source6.point_source_magnitude("F146")
        assert result2 == [self.source6.source_dict["ps_mag_F146"]]
        result3 = self.source6.point_source_magnitude("z")
        assert result3 == [self.source6.source_dict["ps_mag_z"]]
        result4 = self.source6.point_source_magnitude("y")
        assert result4 == [self.source6.source_dict["ps_mag_y"]]

    def test_ps_magnitude_with_variability(self):
        image_observation_times = np.array([np.pi, np.pi / 2, np.pi / 3])
        result = self.source.point_source_magnitude("r", image_observation_times)
        result_comp = np.array([0.48917028, 0.38842661, 0.27946793])
        npt.assert_almost_equal(result, result_comp, decimal=5)

        image_observation_times2 = np.array([20, 50, 80])
        result2 = self.source6.point_source_magnitude("F146", image_observation_times2)
        result_comp2 = self.source6.variability_class.variability_at_time(
            image_observation_times2
        )
        npt.assert_almost_equal(result2, result_comp2, decimal=5)

    def test_es_magnitude(self):
        result = self.source.extended_source_magnitude("r")
        assert result == [23]
        with pytest.raises(ValueError):
            self.source.extended_source_magnitude("j")

    def test_ps_magnitude_array(self):
        result = self.source2.point_source_magnitude("r")
        assert len(result) == 5

    def test_extended_source_position(self):

        pos = self.source.extended_source_position(
            center_lens=np.array([0.002, -0.002]), draw_area=4 * np.pi
        )
        assert len(pos) == 2
        assert isinstance(pos[0], float)
        assert isinstance(pos[1], float)

    def test_point_source_position_without_offset(self):
        pos = self.source.point_source_position(
            center_lens=np.array([0.002, -0.002]), draw_area=4 * np.pi
        )
        assert len(pos) == 2
        assert isinstance(pos[0], float)
        assert isinstance(pos[1], float)

    def test_point_source_position_with_offset(self):
        pos = self.source2.point_source_position(
            center_lens=np.array([0.002, -0.002]), draw_area=4 * np.pi
        )
        assert len(pos) == 2
        assert isinstance(pos[0], float)
        assert isinstance(pos[1], float)

    def test_kwargs_extended_source_light(self):
        center_lens = np.array([0, 0])
        draw_area = 4 * np.pi
        kwargs = self.source2.kwargs_extended_source_light(
            center_lens, draw_area, band="r"
        )
        assert len(kwargs) == 1
        assert isinstance(kwargs[0], dict)

        kwargs_source11 = self.source11.kwargs_extended_source_light(
            center_lens, draw_area, band="F106"
        )
        kwargs_source11_ref = [
            {
                "R_sersic": 0.1651633078964498,
                "center_x": 0.30298310338567075,
                "center_y": -0.3505004565139597,
                "e1": -0.06350855238708408,
                "e2": -0.08420760408362458,
                "magnitude": 21.434711611915137,
                "n_sersic": 1.0,
            }
        ]
        assert kwargs_source11 == kwargs_source11_ref

    def test_kwargs_extended_source_light_double_sersic(self):
        center_lens = np.array([0, 0])
        draw_area = 4 * np.pi
        kwargs = self.source3.kwargs_extended_source_light(
            center_lens, draw_area, band="i"
        )
        assert len(kwargs) == 2
        assert all(isinstance(kwargs_item, dict) for kwargs_item in kwargs)
        with pytest.raises(ValueError):
            self.source6.kwargs_extended_source_light(center_lens, draw_area, band="i")
        with pytest.raises(ValueError):
            self.source5.kwargs_extended_source_light(center_lens, draw_area, band="i")
        with pytest.raises(ValueError):
            self.source10.kwargs_variability_extracted
        with pytest.raises(ValueError):
            self.source9.kwargs_variability_extracted
        with pytest.raises(ValueError):
            self.source8.kwargs_variability_extracted
        assert self.source7.kwargs_variability_extracted["r"]["ps_mag_r"] == 17
        assert self.source7.kwargs_variability_extracted["r"]["MJD"] == 20

    def test_source_agn(self):

        obs_time = 20
        later_obs_time = 50

        # Make sure both "g" and "y" band magnitudes exist, and they should
        # not be equal
        g_mag = self.source_agn_1.point_source_magnitude(
            "g", image_observation_times=obs_time
        )
        y_mag = self.source_agn_1.point_source_magnitude(
            "y", image_observation_times=obs_time
        )
        assert g_mag != y_mag

        # Show that the light curves evolve with time
        g_mag_later_time = self.source_agn_1.point_source_magnitude(
            "g", image_observation_times=later_obs_time
        )
        assert g_mag != g_mag_later_time

        # Create a second source and show it has a different magnitude
        g_mag_2 = self.source_agn_2.point_source_magnitude(
            "g", image_observation_times=later_obs_time
        )
        assert g_mag_2 != g_mag

        # Test errors
        with pytest.raises(ValueError):
            self.source_agn_error.point_source_magnitude(
                "g", image_observation_times=obs_time
            )
        with pytest.raises(ValueError):
            self.source_agn_bpl_error.point_source_magnitude(
                "g", image_observation_times=obs_time
            )
        with pytest.raises(ValueError):
            self.source_agn_error_no_cosmo.point_source_magnitude(
                "g", image_observation_times=obs_time
            )

        with pytest.raises(ValueError):
            self.source_agn_error_no_magnitude.point_source_magnitude(
                "g", image_observation_times=obs_time
            )

        with pytest.raises(ValueError):
            self.source_agn_error_agn_not_in_kwargs.point_source_magnitude("r")

        # Create a source with a broken power law
        broken_power_law_time_1 = self.source_bpl_agn.point_source_magnitude(
            "i", image_observation_times=obs_time
        )
        # Show bpl also evolves with time
        broken_power_law_time_2 = self.source_bpl_agn.point_source_magnitude(
            "i", image_observation_times=later_obs_time
        )
        assert broken_power_law_time_1 != broken_power_law_time_2

    def test_add_mean_mag_to_source_table(self):
        final_source_table = add_mean_mag_to_source_table(
            self.source_dict3, [23, 24, 25], ["i", "r", "g"]
        )
        assert final_source_table["ps_mag_i"] == 23
        assert final_source_table["ps_mag_r"] == 24
        assert final_source_table["ps_mag_g"] == 25

    def test_extended_source_light_model(self):
        light_model1 = self.source_light_model_1.extended_source_light_model()
        light_model2 = self.source_light_model_2.extended_source_light_model()
        assert light_model1[0] == "SERSIC_ELLIPSE"
        assert len(light_model2) == 2
        assert light_model2[0] == "SERSIC_ELLIPSE"
        with pytest.raises(ValueError):
            self.source_light_model_3.extended_source_light_model()

    def test_surface_brightness_reff(self):
        source_dict = {
            "angular_size": 1,
            "center_x": 0.30298310338567075,
            "center_y": -0.3505004565139597,
            "e1": 0.06350855238708408,
            "e2": -0.08420760408362458,
            "mag_g": 15,
            "n_sersic": 1.0,
            "z": 3.123,
        }

        band = "g"
        source = Source(source_dict=source_dict)
        mag_arcsec2 = source.surface_brightness_reff(band=band)
        npt.assert_almost_equal(mag_arcsec2, 16.995, decimal=2)


if __name__ == "__main__":
    pytest.main()
