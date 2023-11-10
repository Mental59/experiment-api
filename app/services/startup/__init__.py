from ..file_manager import artifact_manager

def on_startup():
  artifact_manager.create_artifact_folders()
