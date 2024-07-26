from setuptools import setup

setup(
    name="platerec",
    version="0.0.2",
    packages=["platerec"],
    include_package_data=True,
    url="https://github.com/pstwh/platerec",
    keywords="plate, ocr, read, brazilian",
    package_data={"platerec": ["artifacts/*.onnx"]},
    python_requires=">=3.5, <4",
    install_requires=["pillow==10.4.0", "opencv-python-headless==4.10.0.84", "platedet==0.0.2"],
    extras_require={
        'cpu': [
            'onnxruntime==1.18.1',
        ],
        "gpu": [
            "onnxruntime-gpu==1.18.1",
        ],
    },
    entry_points={
        "console_scripts": ["platerec=platerec.cli:main"],
    },
    description="Read license plates",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
