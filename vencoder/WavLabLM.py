import torch
from s3prl.nn import S3PRLUpstream

from vencoder.encoder import SpeechEncoder


class WavLabLM(SpeechEncoder):  
    def __init__(self, vec_path="", device=None):
        super().__init__()
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        model = S3PRLUpstream("wavlablm_ms_40k")
        model.eval()
        self.hidden_dim = 1024
        self.model = model.to(self.dev)

    def encoder(self, wav):
        with torch.no_grad():
            wavs = wav.view(1, -1) 
            wavs_len = torch.LongTensor([wav.shape[0]])
            hs, _ = self.model(wavs.to(wav.device), wavs_len.to(wav.device))
            return hs[0].transpose(1, 2).to(self.dev)