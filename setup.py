from setuptools import setup

setup(name='amore',
      version='0.1',
      description='Automated Morality Extraction (AMorE) for Python',
      url='https://github.com/medianeuroscience/amore',
      author='Anonymized.',
      author_email='fhopp@ucsb.edu',
      license='MIT',
      packages=['amore'],
      scripts=['bin/amore'],
      package_data={'':['dictionaries/*', 'template_input.csv']},
      include_package_data=True, 
      install_requires=[
          'pandas',
          'progressbar2',
          'spacy',
          'sklearn',
          'nltk',
          'numpy'
      ],
      zip_safe=False)
