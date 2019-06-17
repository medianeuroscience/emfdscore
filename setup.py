from setuptools import setup
from setuptools.command.install import install
from subprocess import check_call

class LoadSpacyLanguage(install):
    """Ensures that spaCy language model is loaded"""
    def run(self):
        check_call("python -m spacy download en".split())
        install.run(self)


setup(name='amore',
      version='0.1',
      description='Automated Morality Extraction (AMorE) for Python',
      url='https://github.com/medianeuroscience/amore',
      author='Anonymized.',
      author_email='fhopp@ucsb.edu',
      license='MIT',
      packages=['amore'],
      scripts=['bin/amore'],
      include_package_data=True, 
      install_requires=[
          'pandas',
          'progressbar2',
          'spacy',
          'sklearn',
          'nltk',
          'numpy'
      ],
      cmdclass={'install': LoadSpacyLanguage},
      zip_safe=False)