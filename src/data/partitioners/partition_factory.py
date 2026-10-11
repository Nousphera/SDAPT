
from data.partitioners.dirichlet_partitioner import DirichletDataPartitioner
from data.partitioners.few_shot_partitioner import FewShotDataPartitioner
from data.partitioners.micro_domain_partitioner import MicroDomainPartitioner
from data.partitioners.random_partitioner import RandomPartitioner

class PartitionFactory:
    @staticmethod
    def create(partitioner_type):
        if partitioner_type == "random":
            return RandomPartitioner
        elif partitioner_type == "reduce_sample":
            return FewShotDataPartitioner
        elif partitioner_type == "few_shot":
            return FewShotDataPartitioner
        elif partitioner_type == "micro_domain":
            return MicroDomainPartitioner
        elif partitioner_type == "dirichlet":
            return DirichletDataPartitioner
        else:
            raise ValueError(f"Unknown partitioner: {partitioner_type}")