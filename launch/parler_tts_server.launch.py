from launch import launch_description
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return launch_description.LaunchDescription([
        DeclareLaunchArgument(
            'model_name',
            default_value='parler-tts/parler-tts-mini-v1',
            description='Language model'
            
            #英語モデル
                #ミニ           ：parler-tts/parler-tts-mini-v1
                #ミニジェニー   ：parler-tts/parler-mini-v1-jenny
                #感情指定可能   ：parler-tts/parler-tts-mini-expresso
                #CPU向けジェニー：parler-tts/parler-tiny-v1-jenny
            #日本語モデル
                #ミニ           ：2121-8/japanese-parler-tts-mini
        ),
        DeclareLaunchArgument(
            'description',
            default_value="female speaks animated voice, very high pitch, slowly, very clear, high quality audio.",
            description='Description of the speaker voice'
        ),
        Node(
            package='parler_tts_ros',
            executable='parler_tts_server',
            name='parler_tts_action_server',
            parameters=[
                {'model_name': LaunchConfiguration('model_name')},
                {'description': LaunchConfiguration('description')},
            ],
            output='screen'
        ),
    ])

if __name__ == '__main__':
    generate_launch_description()