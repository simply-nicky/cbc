import setuptools

with open('README.md', 'r') as readme:
    long_description = readme.read()

setuptools.setup(
    name="cbc",
    version="1.0.1",
    author="Nikolay Ivanov",
    author_email="simply.i.nicky@gmail.com",
    description="Convergent beam crystallography project",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/simply-nicky/cbc",
    packages=setuptools.find_packages(),
    data_files=[
        ("utils/asf", ["cbc/utils/asf/asf_henke.npy", "cbc/utils/asf/asf_waskif.npy"])
    ],
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent"
    ],
    python_requires='>=2.7'
)