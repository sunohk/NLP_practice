import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm

class RnnlmGen(Rnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100): 
        '''
        0) self = 해당 메서드가 호출되는 객체 자신을 참조
        1) start_id = 시퀀스를 생성할 때 시작하는 단어의 ID입니다. 이 ID는 시퀀스의 첫 번째 단어로 사용
        2) skip_ids = 생성 중 건너뛸 단어 ID들의 리스트입니다. 이 리스트에 포함된 단어들은 샘플링에서 제외됨
        3) sample_size = 생성할 단어 시퀀스의 길이. 기본값은 100으로, 최대 100개의 단어가 생성된다는 의미
        
        '''
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1,1)
            score = self.predict(x)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))
        
        return word_ids