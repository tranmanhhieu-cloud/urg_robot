from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot_supervisor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament index + package.xml (bắt buộc)
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),

        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'pyyaml',  # để đọc waypoints.yaml
    ],
    zip_safe=True,
    maintainer='tran',
    maintainer_email='manhhieu11022005@gmail.com',
    description='Robot supervisor: commander + voice chat + navigation bridge',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'commander = robot_supervisor.commander_node:main',
            'voice_chat = robot_supervisor.voice_chat_and_nav:main',
        ],
    },
)