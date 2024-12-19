class ParticipantGroup(list[int]):
    pass


class Allocation(list[ParticipantGroup]):
    pass


class AllocationEnsemble(list[Allocation]):
    def calc_n_meetings_alo(self) -> int:
        return sum(
            len(p_stats)
            for p_id, p_stats in self.calc_meetings().items()
        )

    def calc_meetings(self) -> dict[int, dict[int, int]]:
        p_ids = {
            p_id
            for allocation in self
            for group in allocation
            for p_id in group
            if p_id
        }

        meetings = {}
        for p_id in p_ids:
            meetings[p_id] = {}
            for allocation in self:
                for group in allocation:
                    for p_id_other in group:
                        if p_id == p_id_other:
                            continue
                        if p_id_other not in meetings[p_id]:
                            meetings[p_id][p_id_other] = 1
                        meetings[p_id][p_id_other] += 1

        return meetings


class AllocatorResult:
    def __init__(self, ensemble: None | AllocationEnsemble = None):
        self.ensemble = ensemble
