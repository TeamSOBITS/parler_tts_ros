from launch import launch_description
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return launch_description.LaunchDescription([
        DeclareLaunchArgument(
            'model_name',
            default_value='parler-tts/parler-mini-v1',
            description='Language model'
            
            #日本語モデル：2121-8/japanese-parler-tts-mini

            #英語モデル
                #ミニ        ：parler-tts/parler-mini-v1
                #ミニジェニー：parler-tts/parler-mini-v1-jenny

                #感情　　　　：parler-tts/parler-tts-mini-expresso
            
                #CPU向け：parler-tts/parler-tts-tiny-v1
                #CPU向けジェニー：parler-tts/parler-tiny-v1-jenny
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






















































#隠し要素：2121-8/japanese-parler-tts-mini-bate