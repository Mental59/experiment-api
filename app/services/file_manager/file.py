from fastapi import UploadFile


async def write_file(path: str, file: UploadFile):
  content = await file.read()

  with open(path, 'wb') as file:
    file.write(content)
