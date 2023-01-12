import site
site.addsitedir("../ARGUS")

class ARGUS_pnb_roi():
    
    def __init__(self):
        
    def generate_roi(self, vid):
        self.preprocessed_pnb_video = self.preprocess_pnb.process(vid)
        super().preprocess(self.preprocessed_pnb_video)
