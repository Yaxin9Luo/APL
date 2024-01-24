import torch
import torch.nn as nn

class LanguageDecoder(nn.Module):
    def __init__(self, __C):
        super(LanguageDecoder, self).__init__()

        self.hidden_size = __C.HIDDEN_SIZE
        self.feature_size = __C.HIDDEN_SIZE  # 特征维度
        self.num_layers = 1

        # 定义LSTM，以处理视觉嵌入
        self.lstm = nn.LSTM(input_size=self.feature_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

        # 将LSTM的输出转换为文本特征的尺寸
        self.fc = nn.Linear(self.hidden_size, self.feature_size)

    def forward(self, visual_embedding):
        h0 = torch.zeros(self.num_layers, visual_embedding.size(0), self.hidden_size).to(visual_embedding.device)
        c0 = torch.zeros(self.num_layers, visual_embedding.size(0), self.hidden_size).to(visual_embedding.device)

        output, _ = self.lstm(visual_embedding, (h0, c0))

        # 取最后一个时间步的输出作为特征
        last_output = output[:, -1, :]

        # 将LSTM的输出转换为目标特征尺寸
        text_feature = self.fc(last_output)
        return text_feature.unsqueeze(1)

backbone_dict={
    'lstm':LanguageDecoder,
}
def language_decoder(__C):
    lang_enc=backbone_dict[__C.LANG_DEC](__C)
    return lang_enc