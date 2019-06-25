from setuptools import setup

setup(name='emfdscore',
      version='0.0.1',
      description='Extended Moral Foundation Dictionary Scoring for Python',
      url='https://github.com/medianeuroscience/emfdscore',
      author='Anonymized.',
      author_email='fhopp@ucsb.edu',
      license='MIT',
      packages=['emfdscore'],
      scripts=['bin/emfdscore'],
      include_package_data=True, 
      install_requires=[
          'pandas',
          'progressbar2',
          'sklearn',
          'nltk',
          'numpy'
      ],
      zip_safe=False)