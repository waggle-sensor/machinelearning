import setuptools

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setuptools.setup(
    name="ALE",
    version="0.0.1",
    url="https://github.com/kiwidamien/roman",
    author="Enrique Nueve",
    author_email="activelearningexploratorium@gmail.com",
    description="Test framework for Active Learning using TensorFlow2)",
    long_description=open('README.md').read(),
    packages=setuptools.find_packages(),
    install_requires=[REQUIREMENTS],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)