# CodeAlpha AI Projects

This repository contains four independent Artificial Intelligence projects prepared for the CodeAlpha internship task list.

Each folder is a standalone project with its own dependencies, source code, and setup instructions. Create a virtual environment for the task you want to run, install that task's `requirements.txt`, and follow its README.

## Projects

| Folder | Project | Main idea |
| --- | --- | --- |
| `task_1_language_translation` | Language Translation Tool | Translate text between supported languages through Google Translate. |
| `task_2_faq_chatbot` | FAQ Chatbot | Find the closest answer in a topic-specific FAQ collection using TF-IDF and cosine similarity. |
| `task_3_music_generation` | Music Generation with AI | Prepare MIDI notes, train an LSTM, and write a newly generated MIDI file. |
| `task_4_object_tracking` | Object Detection and Tracking | Detect objects with YOLO and maintain simple tracking IDs across video frames. |

## General notes

- Python 3.10 or newer is recommended.
- The first and second projects use Streamlit and open in a browser.
- The third project needs a directory of MIDI files before training.
- The fourth project needs a webcam or a video file. YOLO weights are downloaded automatically by Ultralytics the first time they are used.

The projects are intentionally kept separate so they can be demonstrated and submitted individually.
