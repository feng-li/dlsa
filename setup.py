import os
from setuptools import setup

def read(file):
    return open(os.path.join(os.path.dirname(__file__), file)).read()

setup(name='dlsa',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      version='0.01',
      description='Distributed Least Squares Approximations',
      keywords='spark, spark-ml, pyspark, mapreduce',
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      url='https://github.com/feng-li/dlsa',
      author='Feng Li',
      author_email='feng.li@cufe.edu.cn',
      license='MIT',
      packages=['dlsa'],
      install_requires=[
          'pyspark >= 2.3.1',
          'sklearn >= 0.21.2',
          'numpy   >= 1.16.3',
          'pandas  >= 0.23.4',
          'rpy2    >= 3.0.4',
      ],
      zip_safe=False,
      python_requires='>=3.7',
)
