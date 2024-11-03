class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        #print(f"Adding part: {part}")  # debug
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        full_transcript = ' '.join(self.transcript_parts)
        # print(f"Full transcript_from_transcript_collector: {full_transcript}")  # debug
        return full_transcript


transcript_collector = TranscriptCollector()
