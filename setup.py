from setuptools import setup

package_name = 'local_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=['planner_node', 'preprocess', 'postprocess'],
    package_dir={'': 'scripts'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'config/planner_params.yaml', 'launch/planner_launch.py', 'models/mobilenet_dummy.pt']),
    ],
    install_requires=['setuptools', 'numpy', 'torch'],
    zip_safe=True,
    maintainer='maintainer',
    maintainer_email='example@example.com',
    description='MobileNetV2 local planner.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'planner_node = planner_node:main',
        ],
    },
)
