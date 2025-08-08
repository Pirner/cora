import re
import json


def main():
    response = '{"message":"in construction","outputs":"<think>\n\n</think>\n\n<tool_call>\n{\"name\": \"get_temperature_date\", \"arguments\": {\"location\": \"Nuremberg\", \"date\": \"2024-08-31\", \"unit\": \"celsius\"}}\n</tool_call><|im_end|>"}'
    start = '<tool_call>'
    end = '</tool_call>'
    # tmp = json.loads(response)
    # print(tmp)
    rx = r'{}.*?{}'.format(re.escape(start), re.escape(end))
    for match in re.findall(rx, response, re.S):
        parsed_match = match.removeprefix(start)
        parsed_match = parsed_match.removesuffix(end)
        # parsed_match = parsed_match.replace('\n', '')
        # parsed_match = parsed_match.strip('\n')
        # print(parsed_match)
        tmp = json.loads(parsed_match)
        print(tmp)


if __name__ == '__main__':
    main()
