from src.query_parser.eva_statement import EvaStatement
from src.query_parser.types import StatementType
from src.expression.abstract_expression import AbstractExpression
from src.loaders.abstract_loader import AbstractLoader
from src.query_parser.table_ref import TableRef
from typing import List


class InsertStatement(EvaStatement):
    """
    Insert Statement constructed after parsing the input query
    Attributes
    ----------
    _target_list : List[AbstractExpression]
        list of target attributes in the insert query,
        each attribute is represented as a Abstract Expression
    _to_table : TableRef
        table reference in the select query
    **kwargs : to support other functionality, Orderby, Distinct, Groupby.
    """

    def __init__(self, target_list: List[AbstractExpression] = None,
                 to_table: TableRef = None,
                 **kwargs):
        super().__init__(StatementType.SELECT)
        self._to_table = to_table
        self._target_list = target_list

    
    @property
    def target_list(self):
        return self._target_list
    
    @target_list.setter
    def target_list(self, target_expr_list: List[AbstractExpression]):
        self._target_list = target_expr_list

    @property
    def from_table(self):
        return self._from_table

    @from_table.setter
    def from_table(self, table: TableRef):
        self._from_table = table

    def __str__(self) -> str:
        print_str = "INSERT INTO {} VALUES {}".format(self._target_list,
                                                        self._to_table,
                                                        self._where_clause)
        return print_str
