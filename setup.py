from setuptools import setup, find_packages

def readme(short=False):
    with open('README.rst') as f:
        if short:
            return f.readlines()[1].strip()
        else:
            return f.read()

def get_version_from_readme() -> str:
    """Get current version of package from README.rst"""
    readme_text = readme()
    for line in readme_text.splitlines():
        if ":Version:" in line:
            current_version = line.split(":")[-1].strip()
            return current_version
    raise ValueError("Could not find version in README.rst")

setup(
    name='margarine',
    version=get_version_from_readme(),
    description='margarine: Posterior Sampling and Marginal Bayesian Statistics',
    long_description=readme(),
    author='Harry T. J. Bevins',
    author_email='htjb2@cam.ac.uk',
    url='https://github.com/htjb/margarine',
    packages=find_packages(),
    install_requires=['numpy',
                      'tensorflow; sys_platform != "darwin"',
                      'tensorflow-macos; sys_platform == "darwin"',
                      'tensorflow_probability',
                      'anesthetic', 'scipy', 'pandas',
                      'scikit-learn', 'tqdm'],
    license='MIT',
    extras_require={
          'docs': ['sphinx', 'sphinx_rtd_theme', 'numpydoc'],
          },
    tests_require=['pytest', 'torch'],
    classifiers=[
               'Intended Audience :: Science/Research',
               'Natural Language :: English',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Astronomy',
               'Topic :: Scientific/Engineering :: Physics',
    ],
)
