//! Binary Tensor terminal handlers.

use super::super::Tensor;
use super::{no_parameters, parameter, state};
use crate::{CollectionState, StorageContext};

macro_rules! binary_handler {
    ($($handler:ident => $method:ident, $scalar:ident);* $(;)?) => {
        $(pub(super) struct $handler<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

        impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
            for $handler<'a, Txn>
        {
            fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
            where
                'txn: 'a,
            {
                Some(Box::new(move |txn, mut params| {
                    Box::pin(async move {
                        let right = parameter(&mut params, "r")?;
                        if let Some(scalar) = right.as_number() {
                            no_parameters(&params)?;
                            return Ok(state(self.0.$scalar(txn.id(), scalar).await?));
                        }

                        let right = right.into_tensor()?;
                        no_parameters(&params)?;
                        Ok(state(self.0.$method(txn.id(), &right).await?))
                    })
                }))
            }
        })*
    };
}

binary_handler! {
    Add => add, add_scalar;
    Sub => sub, sub_scalar;
    Mul => mul, mul_scalar;
    Div => div, div_scalar;
    Pow => pow, pow_scalar;
    Rem => rem, rem_scalar;
    Log => log, log_scalar;
    Eq => eq, eq_scalar;
    Ne => ne, ne_scalar;
    Gt => gt, gt_scalar;
    Ge => ge, ge_scalar;
    Lt => lt, lt_scalar;
    Le => le, le_scalar;
    And => and, and_scalar;
    Or => or, or_scalar;
    Xor => xor, xor_scalar;
}

pub(super) struct Cond<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Cond<'a, Txn>
{
    fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, mut params| {
            Box::pin(async move {
                let then = parameter(&mut params, "then")?.into_tensor()?;
                let or_else = parameter(&mut params, "or_else")?.into_tensor()?;
                no_parameters(&params)?;
                Ok(state(self.0.cond(txn.id(), &then, &or_else).await?))
            })
        }))
    }
}
