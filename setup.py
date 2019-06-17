from setuptools import setup

setup(name='pyamore',
      version='0.0.4',
      description='Automated Morality Extraction (AMorE) for Python',
      url='https://github.com/medianeuroscience/pyamore',
      author='Anonymized.',
      author_email='fhopp@ucsb.edu',
      license='MIT',
      packages=['pyamore'],
      scripts=['bin/pyamore'],
      include_package_data=True, 
      install_requires=[
          'pandas',
          'progressbar2',
          'sklearn',
          'nltk',
          'numpy'
      ],
      zip_safe=False)