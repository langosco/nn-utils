from setuptools import setup

setup(name='nn-utils',  # For pip. E.g. `pip show`, `pip uninstall`
      version='0.0.1',
      author="Lauro Langosco",
      description="Common useful things for neural network training.",
      packages=["nn_utils"], # For python. E.g. `import python_template`
      install_requires=[
          "numpy",
          ],
      )
