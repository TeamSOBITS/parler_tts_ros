from setuptools import find_packages, setup

package_name = 'parler_tts_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sobits',
    maintainer_email='sobits@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'parler_tts_server = parler_tts_ros.parler_tts_server:main',
            'parler_tts_client = parler_tts_ros.parler_tts_client:main',
        ],
    },
)
