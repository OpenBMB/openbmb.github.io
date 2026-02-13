// Omni Case Data - Video demos for MiniCPM-o 4.5
const DEMO_DATA = {
  "meta": {
    "title": "MiniCPM-o 4.5",
    "description": {
      "zh": "端侧全模态实时交互模型 · Omni 能力展示",
      "en": "End-side Omni-modal Real-time Interaction Model · Omni Capability Showcase"
    },
    "version": "1.0.0",
    "disclaimer": {
      "zh": "免责声明：本页面展示的视频案例仅用于AI技术演示目的。所有视频中的 AI 回复均为模型实时生成，不代表任何真实个人的观点、立场或意图。视频内容可能包含模型的不完美表现，仅供技术参考。",
      "en": "Disclaimer: The video demonstrations on this page are for AI technology showcase purposes only. All AI responses in the videos are generated in real-time by the model and do not represent the views, positions, or intentions of any real individual. Video content may include imperfect model performance and is provided for technical reference only."
    }
  },
  "abilities": [
    {
      "id": "omni_fullduplex",
      "name": {
        "zh": "实时全模态全双工对话",
        "en": "Omni Full-duplex Multimodal Live Streaming"
      },
      "description": {
        "zh": "实时全模态对话中「边看、边听、边说」",
        "en": "See, listen, and speak simultaneously in a real-time omnimodal conversation"
      },
      "sub_abilities": [
        {
          "id": "visual_proactive_reminding",
          "name": {
            "zh": "视觉主动提醒",
            "en": "Visual Proactive Reminding"
          },
          "case_tags": [
            {
              "id": "elevator_arrival",
              "name": {
                "zh": "电梯到达提醒",
                "en": "Elevator Arrival Reminding"
              },
              "cases": [
                { "id": "proactive_visual_01", "video": "videos/proactive_visual_01.mp4" }
              ]
            },
            {
              "id": "subway_arrival",
              "name": {
                "zh": "地铁到站提醒",
                "en": "Subway Arrival Reminding"
              },
              "cases": [
                { "id": "proactive_visual_02", "video": "videos/proactive_visual_02.mp4" }
              ]
            }
          ]
        },
        {
          "id": "audio_proactive_reminding",
          "name": {
            "zh": "声音主动提醒",
            "en": "Audio Proactive Reminding"
          },
          "case_tags": [
            {
              "id": "air_fryer_ding",
              "name": {
                "zh": "空气炸锅叮了提醒",
                "en": "Air Fryer Ding Reminding"
              },
              "cases": [
                { "id": "proactive_audio_01", "video": "videos/proactive_audio_01.mp4" }
              ]
            }
          ]
        },
        {
          "id": "immersive_dialogue",
          "name": {
            "zh": "沉浸感对话",
            "en": "Immersive Dialogue"
          },
          "case_tags": [
            {
              "id": "shopping_mall",
              "name": {
                "zh": "陪逛商场",
                "en": "Shopping Mall Companion"
              },
              "cases": [
                { "id": "realtime_immersive_01", "video": "videos/realtime_immersive_01.mp4" }
              ]
            },
            {
              "id": "street_walk",
              "name": {
                "zh": "陪街道散步",
                "en": "Street Walk Companion"
              },
              "cases": [
                { "id": "realtime_immersive_02", "video": "videos/realtime_immersive_02.mp4" }
              ]
            }
          ]
        },
        {
          "id": "realtime_captioning_memory",
          "name": {
            "zh": "实时描述和记忆",
            "en": "Real-time Captioning and Memory"
          },
          "case_tags": [
            {
              "id": "building_blocks",
              "name": {
                "zh": "拼积木",
                "en": "Building Blocks"
              },
              "cases": [
                { "id": "realtime_captioning_01", "video": "videos/realtime_captioning_01.mp4" }
              ]
            }
          ]
        },
        {
          "id": "omni_voice_clone",
          "name": {
            "zh": "Omni 音色克隆",
            "en": "Omni Voice Clone"
          },
          "case_tags": [
            {
              "id": "nezha_timbre",
              "name": {
                "zh": "哪吒音色",
                "en": "Nezha Timbre"
              },
              "cases": [
                { "id": "voice_clone_nezha", "video": "videos/voice_clone_nezha.mp4", "ref_audio": "videos/timbre/nezha_ref.wav" }
              ]
            },
            {
              "id": "bajie_timbre",
              "name": {
                "zh": "八戒音色",
                "en": "Bajie Timbre"
              },
              "cases": [
                { "id": "voice_clone_bajie", "video": "videos/voice_clone_bajie.mp4", "ref_audio": "videos/timbre/bajie_ref.wav" }
              ]
            },
            {
              "id": "wukong_timbre",
              "name": {
                "zh": "悟空音色",
                "en": "Wukong Timbre"
              },
              "cases": [
                { "id": "voice_clone_wukong", "video": "videos/voice_clone_wukong.mp4", "ref_audio": "videos/timbre/wukong_ref.wav" }
              ]
            },
            {
              "id": "spongebob_timbre",
              "name": {
                "zh": "海绵宝宝音色",
                "en": "SpongeBob Timbre"
              },
              "cases": [
                { "id": "voice_clone_spongebob", "video": "videos/voice_clone_spongebob.mp4", "ref_audio": "videos/timbre/spongebob_ref.wav" }
              ]
            }
          ]
        }
      ]
    }
  ]
};
