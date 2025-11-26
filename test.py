from langchain_community.document_loaders import YoutubeLoader

def get_youtube_transcript(video_id: str):
    try:
        loader = YoutubeLoader.from_youtube_url(
            f"https://www.youtube.com/watch?v={video_id}",
            add_video_info=False
        )
        docs = loader.load()

        transcript = "\n".join([doc.page_content for doc in docs])
        return transcript
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Example usage:
video_id = "j4gqPboA3Ew"   # Replace with your ID
transcript = get_youtube_transcript(video_id)
print(transcript)
