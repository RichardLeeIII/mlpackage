import pandas as pd
import numpy as np 
from pathlib import Path
import os
import sys

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import pandas as pd
import numpy as np
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, save_pipeline
import prediction_model.processing.preprocessing as pp
import prediction_model.pipeline as pipe

def perfrom_training():
    train_data=load_dataset(config.TRAIN_FILE)
    train_y=train_data[config.TARGET].map({'N':0,'Y':1})
    pipe.classification_pipeline.fit(train_data[config.FEATURES],train_y)
    save_pipeline(pipe.classification_pipeline)

if __name__=='__main__':
    perfrom_training()

'''
if __name__=='__main__': perfrom_training() 구문은 파이썬 코드의 진입점을 정의하는 역할을 합니다. 
이를 이해하려면 파이썬 스크립트의 실행 방법에 대해 알아야 합니다.

모듈로 실행될 때: 해당 파이썬 파일이 다른 파이썬 코드에 의해 임포트될 때, 
__name__ 변수는 모듈 이름을 가리킵니다. 이 경우, if __name__=='__main__': 블록 안의 코드는 실행되지 않습니다.

스크립트로 실행될 때: 해당 파이썬 파일이 직접 실행될 때, 
__name__ 변수는 '__main__'으로 설정됩니다. 이 경우, if __name__=='__main__': 블록 안의 코드가 실행됩니다.

즉, 이 구문은 해당 파일이 직접 실행될 때에만 perfrom_training() 
함수를 호출하여 훈련 과정을 수행하도록 합니다. 다른 모듈에서 임포트될 경우, 
훈련 과정이 자동으로 실행되지 않습니다.

요약하자면, 이 구문은 코드의 실행을 제어하여, 
직접 실행될 때에만 perfrom_training() 함수를 호출하도록 합니다.
'''