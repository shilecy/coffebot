import operator

# Map supported operators to functions
ops = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
}

def calculate_expression(expr: str):
    try:
        # Very basic safety: only allow digits, spaces, and operators
        allowed_chars = set("0123456789+-*/(). ")
        if not set(expr).issubset(allowed_chars):
            return {"error": "Invalid characters in expression"}

        # Evaluate the expression safely
        result = eval(expr, {"__builtins__": {}}, ops)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
