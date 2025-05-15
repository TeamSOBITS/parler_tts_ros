from launch import launch_description
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return launch_description.LaunchDescription([
        DeclareLaunchArgument(
            'language',
            default_value='en',
            description='Language for TTS (en or ja)'
        ),
        DeclareLaunchArgument(
            'description',
            default_value='Alisa.fast speed. Expression is rich. The speaking voice is very clear.',
            description='Description of the speaker voice'
        ),
        Node(
            package='parler_tts_ros',
            executable='parler_tts_server',
            name='parler_tts_action_server',
            parameters=[
                {'language': LaunchConfiguration('language')},
                {'description': LaunchConfiguration('description')},
            ],
            output='screen'
        ),
    ])

if __name__ == '__main__':
    generate_launch_description()
