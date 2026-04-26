import data_utils


def parse_answer(choices, dataset):
    ret=None
    ret_raw="None"
    if not (choices is None):
        for choice in choices:
            if isinstance(choice, dict):
                response=choice["message"]["content"]
            else:
                response=choice.message.content
            if response is None:
                continue
            if "Final answer:" in response:
                response_split=response.split("Final answer:")
                if len(response_split) >=2:
                    # Segment by the first "Final answer".
                    reasoning=response_split[0]
                    reasoning_split=reasoning.split("###")
                    for r_split in reasoning_split:
                        r_split=r_split.strip()
                        if r_split.startswith("Answer"):
                            r_split=r_split[6:]
                            r_split=r_split.strip(" \n")
                            reasoning=r_split
                            break
                    answer=data_utils.parse_answer(dataset, response_split[1])
                    ret={"reasoning": reasoning, "answer": answer}
                    ret_raw=response
                    break
    if not ret:
        ret={"reasoning": None, "answer": None}

    return ret, ret_raw

def parse_feedback(choices):
    ret=None
    ret_raw="None"
    if not (choices is None):
        for choice in choices:
            if isinstance(choice, dict):
                response=choice["message"]["content"]
            else:
                response=choice.message.content
            if response is None:
                continue
            if "### Feedback" in response:
                response_split=response.split("### ")
                for r in response_split:
                    r_clean=r.strip()
                    if r_clean.startswith("Feedback"):
                        r_clean=r_clean[8:]
                        r_clean=r_clean.strip(' \n')
                        ret=r_clean
                        ret_raw=response
                        break
    
    return ret, ret_raw
