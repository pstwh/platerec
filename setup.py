from setuptools import setup

setup(
    name="platerec",
    version="0.0.3",
    packages=["platerec"],
    include_package_data=True,
    url="https://github.com/pstwh/platerec",
    keywords="plate, ocr, read, alpr",
    package_data={"platerec": ["artifacts/*.onnx", "artifacts/*.json"]},
    python_requires=">=3.5, <4",
    install_requires=[
        "pillow==10.4.0",
        "opencv-python-headless==4.10.0.84",
        "platedet==0.0.2",
    ],
    extras_require={
        "cpu": [
            "onnxruntime==1.20.1",
        ],
        "gpu": [
            "onnxruntime-gpu==1.20.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "platerec=platerec.cli:main",
            "platerec-video=platerec.inference_video:main",
            "platerec-image=platerec.inference_image:main",
        ],
    },
    description="Read license plates",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
