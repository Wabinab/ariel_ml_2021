# ariel_ml_2021
Ariel ML Data Challenges 2021 available here: https://www.ariel-datachallenge.space/ML/documentation/description. The challenge is held by ECML-PKDD with other acknowledgements present on the website (link above). 

Note the following have been summarized from the above website. All credits goes to the website and their creators. This page intents to preserve information of the website after the webpage is down in the future. 

[![Run Python Tests](https://github.com/Wabinab/ariel_ml_2021/actions/workflows/pytest.yaml/badge.svg?branch=main)](https://github.com/Wabinab/ariel_ml_2021/actions/workflows/pytest.yaml)


% ---------------------------------------------------------
## Initial thoughts (For Rachel): 
For the data:
Data is very messy and some data like 0001_01_01 seems to be just flat lines without anything useful. Might need to find some way to find a good data entry and then visualize it to get an idea how to clean the data. Note that initial data cleaning had been done by the data preparation team. I don't think we can just throw everything into a model as we might want to see how to arrange it into a csv format for representation, or how to read the file with "flow_from_directory" as the data is too huge to be loaded into Colab's RAM. Though alternatively we could get a "highmem" VM in Google/Azure/etc platform to load everything onto memory. 

For the model: 
Just like the data challenge team is creating a Baseline, it might be a good idea to start with our own Baseline as well and then we could improve from there. The most important thing is to see if it works or not. Then only start tackling on the problems, then only try out different models, then only training on larger datasets (or full datasets depending on the time). 

% ---------------------------------------------------------

### Introduction
Trying to differentiate signals coming from stars, planets or instruments is a challenge. For example, star-spots vs exoplanets' atmosphere. This is very hard if not impossible to solve with conventional astrophysics method, hence machine learning comes into place. 


### Type of Learning Task: 
Supervised, multi-target regression. 

**Features**: 55 noisy light curves corresponding to 55 different wavelengths, each with 300 timesteps of data. The **goal is to predict a set of 55 relative radii (one per wavelength) for any given datapoint**. This is similar to the "parameter" files observed. 

When a planet transit in front/behind a star, assuming the planet is asteroidal (no atmosphere, totally rock) and the star is perfect (no star-spots), only when transitioning in front of the star and when the planet behind the star will there be a dip in the lightcurves (i.e. dip in brightness with time). At other times, the total light detected with a perfect instrument is the total starlight + reflected light from the surface of the planet in the instrument's direction. See here: https://www.youtube.com/watch?v=RrusIZaWDW8&ab_channel=ExploreAstro

However we know that planets are not perfect, and they contains atmosphere. Still considering a perfect star and instrument, the atmosphere will translate to **different frequencies of light passing through** to reach our instrument, hence different frequencies will be translated to different brightness (larger/smaller dip in lightcurve). The other light are absorbed by the atmosphere. Depending on what the atmosphere is made up of, the frequencies detected by the instrument also differs. The dip is called "transit depth". This is important to work out the chemistry, temperature, cloud coverage, wind speeds, climate, etc of the planet from far away. Note that **this dip occurs at a particular wavelength**, and we could see it if we *plot $R_p/R_\*)^2$ against wavelength*. 

Then the **second noise comes from star spots**. These star spots add noise to the lightcurve and hence render great difficulty in detecting the atmospheric spectrum of the planet. We would like to remove these spots. However, since the star is so far away from us **we cannot predict where the spots are located, and worse, whether the star have spots**. Our own Sun have a star spots period of 11 (some says 22) years and these stars might also have these periods and hence we could not pre-determine the star spots locations. There is little information to help correct spot signatures with astrophysical technique, hence requires Machine Learning. 

However we know that **spot amplitudes are more pronounced in the shorter wavelength (visible light region) than longer (infra-red (IR)) region**. This means scanning in the IR region reduce spots. We could have taken the difference betwen visible and infrared, however: 1° The shape of lightcurve changes with wavelength. In shorter wavelength: rounded bottom (due to stellar limb-darkening) and infrared is more square. 2) In IR region there is also less star light, causing noise on lightcurve to increase significantly (from other light sources which we do not want). 

Third is **Instrumental noise: challenges in detrending the instrument systematic noise (non-Gaussian noise components)**. This could be jitter noise (when floating in space is not always perfectly secured), ranging to quantum physics effects of infrared detectors. Two main effects is **detector persistence and non-Gaussian red-noise**. Persistence is time dependent trends, and is flux dependent, varies with wavelength and depends on what the previous observation has been. Red noise (1/f noise) is determined by other effects such as temperature fluctuations, vibrations. If we do Fourier Transform of the noise, if we plot amplitude against frequency, if the graph is more "exponential decrement" or "logarithmic decrement" then it is more easy problem to detrend than if the line is more "linear". This means the latter could corrup the transit depth measurement. 

#### Description in Parameter files: 
sma = semimajor axis
incl: inclination

Scoring system (to be included due to giant latex lazy to type and lazy to use MathPix). 
