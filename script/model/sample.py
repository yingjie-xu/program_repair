import torch
import torch.nn as nn
from data_clean import inputTensor
from constant import all_letters, n_letters


class Sample:
    def __init__(self):
        self.model = None

    def sample(self, codes):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        with torch.no_grad():  # no need to track history in sampling
            hidden = self.model.initHidden(1).to(device)
            cell = self.model.initCell(1).to(device)
            output = None

            for i in codes:
                it = inputTensor(i).to(device)
                output, hidden, cell = self.model(
                    it[0].unsqueeze(0), hidden, cell)
                topv, topi = output.topk(1)
                if topi == n_letters - 1:
                    return 'END'
                else:
                    letter = all_letters[topi-1]
                    output = letter
            return output

    def test_hw0(self):
        print("====testing hw0====")
        self.model = torch.load('./res/hw0_batch_10/model_at_39000.pt')
        print(self.sample("let distance_tests = [ ( ((0, 0), (3"))

    def test_hw2(self):
        print("====testing hw2====")
        self.model = torch.load('./res/hw2_batch_10/model_at_23000.pt')
        print(self.sample(
            "let partition (p : 'a -> bool) (l : 'a list) : ('a list * 'a list) = let split v (l1, l2) = if p v then (v::l1, l2) else (l1"))
        print(self.sample(
            "let partition (p : 'a -> bool) (l : 'a list) : ('a list * 'a list) = let split v (l1, l2) = if p v then (v::l1"))
        print(self.sample(
            "let make_manager (masterpass : masterpass) : pass_manager = let ref_list : (address * password) list ref = ref [] in let ref_mpass = ref masterpass in in in let helper_verify (p : password) : bool = if !fail >= 3 then raise AccountLocked else if !fail < 3 then if p = !ref_mpass then (fail := 0; incr counter; true) else (incr fail; if !fail < 3 then raise WrongPassword else raise AccountLocked) else raise AccountLocked in let save masterpass addr pass = if helper_verify masterpass then ( (ref_list := (addr, encrypt masterpass pass) :: !ref_list); ) in let get_force masterpass addr = find_map (fun (a,p) -> if addr = a then Some(decrypt masterpass p) else None) !ref_list in let get masterpass addr = if helper_verify masterpass then get_force masterpass addr else None in let update_master curr_pass new_pass = if helper_verify curr_pass then ( ref_mpass := new_pass; ref_list := List.map (fun (a,p) -> (a"))


if __name__ == '__main__':
    test = Sample()
    test.test_hw0()
    test.test_hw2()
