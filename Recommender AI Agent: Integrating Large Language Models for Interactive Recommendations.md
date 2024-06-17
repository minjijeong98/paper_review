# Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations 
- paper link: https://arxiv.org/abs/2308.16505
- related code: https://github.com/microsoft/RecAI/blob/main/InteRecAgent/README.md


# Paper Summary
## 1 Introduction
- LLM을 활용한 interactive 대화형 추천 시스템인 **InteRecAgent**을 제안
	- LLM은 두뇌, 도메인 내 추천모델은 Tool
- Background

| 모델  | 추천모델                             | LLM                                 |
| --- | -------------------------------- | ----------------------------------- |
| 장점  | 도메인별 item 추천 <br>(사용자 행동 데이터 이용) | 일반 인공지능의 면모<br>general world 지식에 강함 |
| 단점  | 설명, 대화 등 다양한 작업은 어려움             | 도메인 특정적 지식은 부족                      |
| 대안  | **InteRecAgent**                     | 도메인에 맞게 fine-tuning -> 비경제적         |


## 2 Methodology
**InteRecAgent** (interactive Recommender Agent)
- 기본 idea : LLM과 기존 추천 tool (querying, retrieval, ranking)의 결합
- LLM 활용 시 발생 가능 문제에 대한 솔루션

| 문제                            | 솔루션                                   |
| ----------------------------- | ------------------------------------- |
| 입력 프롬프트 길이 제한                 | candidate bus                         |
|                               | long-term and short-term user profile |
| 결과 품질 위해 적절한 demo 사용 필수       | dynamic demonstration                 |
| instruction 이탈, hallucination | reflection                            |

**Overview**
- 일상적인 대화와 item 추천 모두 아우름
- 유저는 LLM과 자연어로 소통
- LLM: 사용자 의도 구문 분석, tool planning (+현재 대화에 tool 필요한지 판단)
  - 일상 대화: LLM의 자체 지식 기반 응답
  - 도메인 내 추천: LLM은 일련의 도구 호출 후, 다음 도구의 실행 결과 관찰해 응답 생성
- 호출의 모든 단계를 한 번에 생성하고 실행 계획을 엄격하게 따라 작업을 수행

![image](https://github.com/minjijeong98/paper_review/assets/162319450/01ad1c3a-5f80-4a7d-adc5-b423bbbd0cbf)

**세부 알고리즘**
#### Dynamic Demonstration
- 현재 사용자 의도와 가장 유사한 몇 가지 데모만 프롬프트에 통합하는 전략
- sentence-transformer 사용해 demo를 벡터로 인코딩, ChromaDB 사용해 저장
- step
  1. 시드 데모 제작: 몇 가지 일반적인 사용자 의도와 그에 따른 execution을 수동으로 작성
  2. LLM 사용해 더 많은 데모 생성: input-first, output-first 전략 사용
     - input-first 전략
       1. LLM이 seed demonstration의 의도 경쟁시켜서 $x$ 생성
       2. 이러한 의도에 대한 계획 $\boldsymbol{p}$ 생성
     - output-first 전략
       1. LLM에 계획 $\boldsymbol{p}$ 제공, 그에 해당하는 사용자 의도 $x$ 생성
       2. LLM 사용해 의도에 대한 계획 $\tilde{\boldsymbol{p}}$ 생성
       3. 생성된 계획 $\tilde{\boldsymbol{p}}$가 주어진 계획 $\boldsymbol{p}$와 일치하는지 검증
          - 불일치: 생성된 의도의 품질이 충분히 높지 않음 의미 -> 제거. 일관성 있는 데모만 유지함
          - 사용 가능한 모든 계획에 해당하는 데모 얻을 수 있으므로 데모에 다양성 제공 가능
- (예시) 게임 도메인에서 생성된 demonstration

| Intent (by GPT-4) | 내가 ITEM1, ITEM2, ITEM3을 선호할 때, 이에 따라 TYPE1과 TYPE2의 아이템을 몇가지 제안해줘.                                                                                                             |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Plan (by GPT-4)   | 1. SQL Retrieval Tool : TYPE1, TYPE2<br>2. Ranking Tool : ITEM 1, 2, 3에 대한 선호 사용<br>3. Candidate Fetching Tool                                                                |
| Plan              | 1. Candidates Storing Tool : ITEM 1, 2, 3<br>2. SQL Retrieval Tool : TYPE<br>3. ItemCF Retrieval Tool: ITEM <br>4. Ranking Tool: preference 이용 <br>5. Candidate Fetching Tool |
| Intent (by GPT-4) | 나는 ITEM1, ITEM2, ITEM3이라는 아이템 목록을 가지고 있다. 나는 ITEM과 유사한 TYPE의 아이템을 원하고, 내 선호에 따라 이들의 순위를 매겨라.                                                                                  |

#### Generate Plan
- step-by-step 전략의 한계
	- retrieval tool이 item 많이 반환 가능 -> LLM 프롬프트 지나치게 길어짐 (-> LLM 성능 저하)
	- LLM이 작업 완료하기 위한 tool 잘못 사용할 수 있음 (잘못된 도구, 주요 실행단계 생략 등)
- **plan-first execution (계획 우선) 전략**
	- 사용자 의도에 따라 tool 호출의 모든 단계 한번에 생성(plan) -> 계획 엄격하게 따라 작업 수행
	- 계획 $\boldsymbol{p}^t = \{p_1^t, \cdots, p_n^t\} = \text{plan}(x^t, C^{t-1}, \mathcal{F}, \mathcal{D}_{x^t})$
		- $x^t$: 사용자의 현재 입력
		- $C^{t-1}$: 대화 컨텍스트 
		- $\mathcal{F}$: 다양한 도구에 대한 설명
		- $D_{x^t}$: 맥락 내 학습을 위한 demo
		- $p_k^t = (f_k, i_k)$,       $f_k$: tool, $i_k$: 그 input
	- 장점 1. LLM의 추론 비용 낮춤
	- 장점 2. Dynamic demo 전략과 원활히 통합 -> 계획 생성 품질 향상

#### Tool Execution
- 계획  $\boldsymbol{p}^t$에 따라 단계별로 도구를 호출하고 각 tool에서 출력을 얻음
	- $\boldsymbol{o}^t = \{o_1^t, \cdots, o_n^t\} = \text{exec}(\boldsymbol{p}^t, \mathcal{F})$,   $o_k^t$: 각 tool $f_k$의 output
	- 마지막 tool의 output에서 얻은 item 정보 $o_n^t$ -> 응답 $y^t$ 생성 위한 LLM의 observation

![그림1](https://github.com/minjijeong98/paper_review/assets/162319450/866de6ef-6c32-49f7-a2cb-9928e4b8a292)
1. Information Query
	- 사용자 질의(*"이 게임의 출시일은 언제이고 가격은 얼마인가요?"*) -> item information query module -> 백엔드 DB에서 자세한 아이템 정보 검색
	- SQL (SQLite)
1. Item Retrieval
	- hard condition: 항목에 대한 명시적인 요구
		- "인기 스포츠 게임을 원해요",  "100달러 미만의 RPG 게임을 추천해 주세요"
		- SQL (SQLite)
	- soft condition: 명시적 속성으로 표현 불가한, 의미적 매칭 모델 사용해야 하는 요구
		-  "콜 오브 듀티 및 포트나이트와 유사한 게임을 원합니다"
		- item-to-item tool (ItemCF): 잠재 임베딩 기반으로 유사한 아이템 매칭
2. Item Ranking
	- 사용자 프로필을 활용하여 선택한 후보에 대한 보다 정교한 사용자 선호도 예측을 실행
	- 추천 항목이 사용자의 즉각적인 의도 뿐만 아니라 사용자의 선호도와도 일치하도록 보장함
	- SASRec

#### Reflection
![image](https://github.com/minjijeong98/paper_review/assets/162319450/1347f24e-0ccd-4a91-a3cf-1fa93824ec1c)
-  self-reflection 중 actor-critic reflection 메커니즘 활용
	- actor: 앞서 결과 생성한 LLM
	- critic: actor의 행동 결정 평가
		1. 판단 $\gamma = \text{reflect}(x^t, C^{t-1}, \boldsymbol{p}^t, \boldsymbol{o}^t, y^t)$ 얻음
  		2. $\gamma>0$ (actor의 실행, 응답 합리적) -> $y^t$ 사용자에게 제공하고 reflection 단계 종료
		3. $\gamma < 0$ (actor의 실행/응답 비합리적) -> actor에게 rechain 지시
- 오류에 대해 견고해짐

---
#### Generate Plan
- **candidate bus**
	- LLM에 많은 item 직접 입력하는건 입력 context 길이 제한으로 무리임
	- candidate bus : 현재의 item 후보를 저장하는 별도의 memory -> 프롬프트에 item 후보 직접 입력할 필요 없음
		- data bus : candidate item 저장 
		- tracker : 각 tool의 output 기록
			- $(f_k, i_k, o_k)$: $k$번째 tool 이름 $f_k$, 남은 후보의 수 $i_k$, 런타임 오류 $o_k$
			- 주요 기능: reflection에서 critic의 판단 도움 ($o^t$의 역할)
		- ![image](https://github.com/minjijeong98/paper_review/assets/162319450/4f8ccaa6-e6f3-4a1b-8cdc-bce7a4311fd1)
			1. 각 대화 턴이 시작될 때 모든 item 포함하도록 후보 item 초기화
			2. 후보 item 읽어서 각 tool 실행하고 업데이트 (순차적)
	- candidate bus 이용해 중간 상태 저장하고 tool 간 커뮤니케이션 용이하게 함

- **long-term and short-term user profile**
	- 사용자의 선호도와 이력을 추적하고 이를 랭킹 도구의 입력으로 활용하여 개인화를 개선
	- 사용자 선호도의 세 가지 측면: liked, disliked, expected
		- liked, disliked: 사용자 선호 (호감, 비호감)
		- expected: 현재 대화 중에 사용자의 즉각적인 요청 모니터링 

## 3 Result
- 실험 데이터셋(3): Steam, MovieLens, Amazon Beauty
- GPT4: 우수한 성능 입증됨 -> "더 작은 언어모델로도 가능할까?"
- RecLlama: 역시 우수한 성능 입증 
	- GPT4 이용해 모방 데이터셋 생성 (instruction, tool execution plan의 쌍) -> 7B 모델 LlaMA2 튜닝

## 4 Review
- 전 과정이 프롬프트 기반으로 수행되는 것으로 보임 -> 비용상 한계는 없는가?
  - Appendix에 제시된 프롬프트 상당히 긺. 모든 과정이 연쇄적인 프롬프트 호출로 구성되는 것으로 논문에서는 보임.
  - 한 명의 사용자에 대해 소요되는 평균 비용은?
  - 대화형 추천 시스템에서는 실시간 대처 중요할텐데, LLM에 온전히 기대는건 시간 오래 걸리지 않을까? 평균 소요 시간은?
- self-refinement 방법으로 충분히 성능 확보 가능한가?
  - LLM이 생성 및 평가 전과정에 개입하고 있음. 얼마나 성능 개선 가능한지 궁금함
  - 보통 몇 회의 rechain 요구되는지? 무한루프의 가능성은 없는가?
- item이 프롬프트가 아닌 candidate bus에서 처리된다고 하는데, 구체적으로 어떻게 구현되는지?






