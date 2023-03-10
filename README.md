#  Asynchronous-Reinforcement-Learning<!-- Yizhou：建议给全称-->

<!---
Yizhou: 一个比较General的建议是，返回值这些可以用英文，e.g. return。 同时我个人理解README最好可以用英文。
-->

<!---
Yizhou: 另一个General的建议，背景之前，几句话描述一下这个repo是做什么的
-->


<!---
Yizhou: 同时请参考 https://www.markdownguide.org/basic-syntax/#code-blocks 完善文档中code的表达
-->
## Background
Asynchronous-Reinforcement-Learning is used for the research of network attack and can be used with [EVO-PopulationBasedTraining](https://github.com/yyzpiero/EVO-PopulationBasedTraining).  Asynchronous-Reinforcement-Learning is based on [Sample-Factory](https://github.com/alex-petrenko/sample-factory), which is based on the APPO (Asynchronous Proximal Policy Optimization) algorithm. Sample-Factory is proposed in the paper "Sample Factory: Egocentric 3D Control From Pixels at 100000 FPS with Asynchronous Reinforcement Learning" by Aleksei Petrenko et al. in 2020. It is a high-throughput training system optimized for a single-machine setting and can achieve throughput higher than 100000 environment frames/second on non-trivial control problems in 3D without sacrificing sample efficiency. Asynchronous-Reinforcement-Learning  transform Sample-Factory (1.23.0) to interface  ,with  the same performance as Sample-factory and can support APPO、A3C、IMPALA.

异步并行算法库接口AsynRL基于2020年Intel和University of Sourhern Californi<!-- Yizhou：cite 作者名字已经其github profile而不是机构名称 -->的论文《Sample Factory: Egocentric 3D Control From Pixels at 100000 FPS with Asynchronous Reinforcement Learning》提出的[Sample Factory](https://github.com/alex-petrenko/sample-factory)，它是为单机设计的高吞吐训练系统，基于APPO(Asynchronous Proximal PolicyOptimization)算法。能在3D控制场景达到100000 FPS。异步并行算法库接口AsynRL主要是将Sample Factory（1.23.0）接口化，性能与Sample-factory相同，支持APPO、A3C、IMPALA三种异步并行算法。

### Installation instructions


<!---
Yizhou: GYM 0.25之后的大改 ，支持不支持？
-->
<!---
Yizhou: 这里可以用表格或者着list
-->
1、Download code: git clone https://github.com/Garytoner/Asynchronous-Reinforcement-Learning.git

2、Create a conda virtual environment： <!-- TYPO -->

Gym：0.17.2

### 安装流程（linux）<!-- 不用 specify是linux？我感觉 -->

1、下载代码: git clone https://github.com/Garytoner/AsynRL.git

2、创建conda虚拟环境：

cd AsynRL

conda env create -f environment.yml

conda activate Asynchronous-Reinforcement-Learning

## Asynchronous-Reinforcement-Learning Instructions

### Interface

model = APPO(env,encoder: str,encodersubtype:str,num_envs_per_worker:int =2,num_workers:int=8,  device: Union[torch.device, str] = "cpu"， policy_kwargs: Optional[Dict[str, Any]] = None)：APPO can be transformed into A3C or IMPALA

### Parameter instructions

param env: training environment

param encoder: encoder type

param encodersubtype: encoding subtype

param num_envs_per_worker: the number of environments that a single actor woreer runs

param num_workers: number of actor workers

param device: device type, if it is cpu, only use cpu, if it is gpu, use both cpu and gpu

param policy_kwargs: other hyperparameters

### Get neural network parameters


<!---
Yizhou: 这里可以用 `model.get_parameters()`
-->

`model.get_parameters()`

return：neural network parameters

### Set neural network parameters

model.set_parameters(parameters)

param parameters:Dictionary type, the key is policy_id, and the value is the corresponding neural network parameter or checkpoint path

### Training

model.train(train_for_env_steps)

param train_for_env_steps:The number of steps for training
<!---
Yizhou: 为什么还要参考main.py,是否可以提供一个例子，比如如何执行可以得到，下面的结果
-->

## Asynchronous-Reinforcement-Learning's train process

1. First create the learner object, and then start the learner process. In the learner process, start the train_loop thread to calculate the gradient and update the network parameters;

2. Create a policy worker object, and then start the policy worker process;

3. Create the actor worker object. When the object is initialized, the actor worker process is created. When the actor worker object is created, two env_runners are created, and the env_runner is responsible for the specific rollout;

4. First reset actor worker, actor_worker sends a request to policy_worker, policy_worker sends a rollout message to actor_worker after receiving the request, actor_worker enters the awake state after receiving it, and then starts rollout;

5. After the actor_worker runs the step, it sends a request to the policy_worker. After the policy_worker receives the request and processes it, it sends an advance_rollout_request message to the actor_worker. After the actor_worker receives the message and processes it, it runs the step and sends a request to the policy_worker. After a complete rollout, it sends a request to the The leraner sends the train command, the learner starts to prepare the buffer, and puts the experience into the buffer, and the train_loop of the learner periodically processes the data and updates the network parameters. The learner puts the network parameters in the shared memory, and the policy worker periodically updates the network parameters from the shared memory.

6. After N steps of training, the training will end. After the actor worker rollout ends, the corresponding actor worker will be suspended. After all the actor workers are suspended, the training will end.

7. After a train is over, the learner puts the network parameters in the shared memory, and the main process updates the relevant network parameters. Network parameters can be set before each train. The set parameters include state_dict and check_point, both of which are overall network parameters.

## training effect and FPS(Frame per Second)：

The figure below shows the training effect and FPS of Asynchronous-Reinforcement-Learning in Atari PongNoFrameSkip-V4 1024 environment, APPO algorithm train 16 times, each 10 million steps of training.
<!---
Yizhou: 这里可以用define一下FPS，like，Frame per Second
-->
### training effect

<p>
    <img src="./images/image3.png"/>
</p>

### FPS

<p>
     <img src="./images/image4.png"/>
</p>

## Sample-Factory

There are three main computational tasks in a typical RL: environment simulation, model inference, and backpropagation. The design principle is to let the slowest task not wait for other tasks, because the throughput of the system depends on the throughput of the slowest task. Each task corresponds to a type of component. Components communicate through high-speed FIFO queues and shared memory. The three components are: 1) Rollout worker: Each rollout worker can have multiple environment instances, and interact with these environments to collect experience. On the one hand, the Rollout worker passes the experience to the policy worker through the shared memory, and on the other hand, to the learner through the shared memory. 2) Policy worker: Collect the status of multiple rollout workers, get actions through policies, and send them back to rollout workers through shared memory. 3) Learner: Get the experience track from the rollout worker through the shared memory, update the policy, and then send it to the policy worker through the GPU memory (because both the policy worker and the learner run on the GPU). The policy worker can generate actions with the new policy.Rollout worker and policy worker are called sampler subsystem together. In order to solve the problem that the rollout worker waits for the policy worker to return the action, the technique of Double-Buffered Sampling is used. That is, a set of environments is stored on the rollout worker and divided into two groups, and the CPU utilization is improved by taking turns.

<p>
    <img src="./images/image1.png"/>
</p>

<p>
    <img src="./images/image2.png"/>
</p>


In this framework, multiple experience trajectories collected from parallel environments in one iteration cannot be digested in one mini-batch by the learner. This will lead to the policy lag problem, that is, the inconsistency between the behavior policy and the target policy. This kind of off-policy learning is more difficult for the PG method. Because the larger the policy lag is, the harder it is to sample from the behavior policy to accurately estimate the gradient. Reducing the number of execution steps in the environment or increasing the mini-batch size can reduce the policy lag. In addition, there are two major techniques used to deal with off-policy learning: trust region (such as PPO clipping) and importance sampling (such as V-trace). Both can be used at the same time. Both are implemented in the Sample Factory.
## Reference
- https://github.com/alex-petrenko/sample-factory.
- https://blog.csdn.net/jinzhuojun/article/details/113796543.
